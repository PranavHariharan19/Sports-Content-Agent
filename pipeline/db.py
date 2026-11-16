from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional

from pydantic import BaseModel

from app.logging_config import get_logger
from schemas.common import (
    CorroborationStatus,
    FactType,
    JobStatus,
    SourceTier,
    SportType,
    StageName,
    VerificationVerdict,
)
from schemas.facts import ConflictRecord, ExtractedFact
from schemas.job import JobRecord, StageRunRecord
from schemas.script import BeatType, GeneratedScript, ScriptBeat
from schemas.sources import StoredSource

logger = get_logger(__name__)

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    sport TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS stage_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    stage_name TEXT NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS llm_cache (
    prompt_hash TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    response_text TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_usage (
    usage_date TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    requests_count INTEGER NOT NULL DEFAULT 0,
    tokens_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (usage_date, provider, model)
);

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    url TEXT NOT NULL,
    domain TEXT NOT NULL,
    source_tier TEXT NOT NULL,
    is_allowlisted INTEGER NOT NULL DEFAULT 1,
    title TEXT,
    cleaned_text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    syndication_cluster_id TEXT,
    is_syndicated INTEGER NOT NULL DEFAULT 0,
    char_count INTEGER NOT NULL DEFAULT 0,
    word_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    claim TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    normalized_entities_json TEXT NOT NULL DEFAULT '[]',
    normalized_value TEXT,
    source_id TEXT NOT NULL,
    evidence_quote TEXT NOT NULL,
    source_tier TEXT NOT NULL,
    published_at TEXT,
    corroborating_sources_json TEXT NOT NULL DEFAULT '[]',
    corroboration_status TEXT NOT NULL,
    requires_attribution INTEGER NOT NULL DEFAULT 0,
    attribution_outlet TEXT,
    is_hedged INTEGER NOT NULL DEFAULT 0,
    is_temporal INTEGER NOT NULL DEFAULT 0,
    is_conflicted INTEGER NOT NULL DEFAULT 0,
    discard_reason TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    entity_or_topic TEXT NOT NULL,
    fact_ids_json TEXT NOT NULL,
    conflicting_values_json TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0,
    chosen_fact_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scripts (
    script_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    headline TEXT NOT NULL,
    beats_json TEXT NOT NULL,
    total_words INTEGER NOT NULL,
    estimated_duration_seconds REAL NOT NULL,
    model_used TEXT NOT NULL,
    verifier_model_used TEXT NOT NULL,
    lint_passed INTEGER NOT NULL,
    all_supported INTEGER NOT NULL,
    is_approved INTEGER NOT NULL DEFAULT 0,
    approved_at TEXT,
    override_logs_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_stage_runs_job_id ON stage_runs(job_id);
CREATE INDEX IF NOT EXISTS idx_llm_cache_provider ON llm_cache(provider, model);
CREATE INDEX IF NOT EXISTS idx_sources_job_id ON sources(job_id);
CREATE INDEX IF NOT EXISTS idx_sources_syndication ON sources(syndication_cluster_id);
CREATE INDEX IF NOT EXISTS idx_facts_job_id ON facts(job_id);
CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(corroboration_status);
CREATE INDEX IF NOT EXISTS idx_conflicts_job_id ON conflicts(job_id);
CREATE INDEX IF NOT EXISTS idx_scripts_job_id ON scripts(job_id);
"""


class DatabaseManager:
    """Manages SQLite connections and queries for the pipeline."""

    def __init__(self, db_path: str | Path = "data/sports_shorts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.info("Initialized database at %s", self.db_path)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a transactional scope around a series of operations."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- Job Operations ---

    def create_job(self, job_id: str, sport: SportType, status: JobStatus = JobStatus.DISCOVERED) -> JobRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, sport, status, created_at, updated_at, error_message)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (job_id, sport.value, status.value, now, now),
            )
        return JobRecord(
            job_id=job_id,
            sport=sport,
            status=status,
            created_at=now,
            updated_at=now,
        )

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT job_id, sport, status, created_at, updated_at, error_message FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                return None
            return JobRecord(
                job_id=row["job_id"],
                sport=SportType(row["sport"]),
                status=JobStatus(row["status"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                error_message=row["error_message"],
            )

    def list_jobs(self, limit: int = 50, sport: Optional[SportType] = None) -> List[JobRecord]:
        with self.get_connection() as conn:
            if sport:
                rows = conn.execute(
                    "SELECT job_id, sport, status, created_at, updated_at, error_message FROM jobs WHERE sport = ? ORDER BY updated_at DESC LIMIT ?",
                    (sport.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT job_id, sport, status, created_at, updated_at, error_message FROM jobs ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [
                JobRecord(
                    job_id=r["job_id"],
                    sport=SportType(r["sport"]),
                    status=JobStatus(r["status"]),
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    error_message=r["error_message"],
                )
                for r in rows
            ]

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None,
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = ?, updated_at = ?, error_message = ?
                WHERE job_id = ?
                """,
                (status.value, now, error_message, job_id),
            )
            return cursor.rowcount > 0

    # --- Stage Run Operations ---

    def record_stage_run(
        self,
        job_id: str,
        stage_name: StageName,
        input_data: Dict[str, Any] | BaseModel,
        output_data: Dict[str, Any] | BaseModel,
        status: str = "COMPLETED",
    ) -> StageRunRecord:
        now = datetime.now(timezone.utc).isoformat()

        input_json = (
            input_data.model_dump_json()
            if hasattr(input_data, "model_dump_json")
            else json.dumps(input_data, ensure_ascii=False)
        )
        output_json = (
            output_data.model_dump_json()
            if hasattr(output_data, "model_dump_json")
            else json.dumps(output_data, ensure_ascii=False)
        )

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO stage_runs (job_id, stage_name, input_json, output_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, stage_name.value, input_json, output_json, status, now),
            )
            stage_id = cursor.lastrowid

        return StageRunRecord(
            id=stage_id,
            job_id=job_id,
            stage_name=stage_name,
            input_json=input_json,
            output_json=output_json,
            status=status,
            created_at=now,
        )

    def get_stage_runs_for_job(self, job_id: str) -> List[StageRunRecord]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, job_id, stage_name, input_json, output_json, status, created_at FROM stage_runs WHERE job_id = ? ORDER BY id ASC",
                (job_id,),
            ).fetchall()
            return [
                StageRunRecord(
                    id=r["id"],
                    job_id=r["job_id"],
                    stage_name=StageName(r["stage_name"]),
                    input_json=r["input_json"],
                    output_json=r["output_json"],
                    status=r["status"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def get_latest_stage_run(self, job_id: str, stage_name: StageName) -> Optional[StageRunRecord]:
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, job_id, stage_name, input_json, output_json, status, created_at
                FROM stage_runs
                WHERE job_id = ? AND stage_name = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (job_id, stage_name.value),
            ).fetchone()
            if not row:
                return None
            return StageRunRecord(
                id=row["id"],
                job_id=row["job_id"],
                stage_name=StageName(row["stage_name"]),
                input_json=row["input_json"],
                output_json=row["output_json"],
                status=row["status"],
                created_at=row["created_at"],
            )

    # --- Source Operations ---

    def save_source(self, source: StoredSource) -> None:
        """Store or update a ground-truth article source in SQLite."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sources (
                    source_id, job_id, url, domain, source_tier, is_allowlisted,
                    title, cleaned_text, text_hash, published_at, fetched_at,
                    syndication_cluster_id, is_syndicated, char_count, word_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source.source_id,
                    source.job_id,
                    source.url,
                    source.domain,
                    source.source_tier.value,
                    1 if source.is_allowlisted else 0,
                    source.title,
                    source.cleaned_text,
                    source.text_hash,
                    source.published_at,
                    source.fetched_at,
                    source.syndication_cluster_id,
                    1 if source.is_syndicated else 0,
                    source.char_count,
                    source.word_count,
                ),
            )
        logger.debug("Stored source %s (%s, %d words) for job %s", source.source_id, source.domain, source.word_count, source.job_id)

    def get_sources_for_job(self, job_id: str) -> List[StoredSource]:
        """Retrieve all stored sources for a job."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT source_id, job_id, url, domain, source_tier, is_allowlisted,
                       title, cleaned_text, text_hash, published_at, fetched_at,
                       syndication_cluster_id, is_syndicated, char_count, word_count
                FROM sources
                WHERE job_id = ?
                ORDER BY is_allowlisted DESC, source_tier ASC
                """,
                (job_id,),
            ).fetchall()
            return [
                StoredSource(
                    source_id=r["source_id"],
                    job_id=r["job_id"],
                    url=r["url"],
                    domain=r["domain"],
                    source_tier=SourceTier(r["source_tier"]),
                    is_allowlisted=bool(r["is_allowlisted"]),
                    title=r["title"],
                    cleaned_text=r["cleaned_text"],
                    text_hash=r["text_hash"],
                    published_at=r["published_at"],
                    fetched_at=r["fetched_at"],
                    syndication_cluster_id=r["syndication_cluster_id"],
                    is_syndicated=bool(r["is_syndicated"]),
                    char_count=r["char_count"],
                    word_count=r["word_count"],
                )
                for r in rows
            ]

    def get_source(self, source_id: str) -> Optional[StoredSource]:
        """Retrieve a specific stored source by ID."""
        with self.get_connection() as conn:
            r = conn.execute(
                """
                SELECT source_id, job_id, url, domain, source_tier, is_allowlisted,
                       title, cleaned_text, text_hash, published_at, fetched_at,
                       syndication_cluster_id, is_syndicated, char_count, word_count
                FROM sources
                WHERE source_id = ?
                """,
                (source_id,),
            ).fetchone()
            if not r:
                return None
            return StoredSource(
                source_id=r["source_id"],
                job_id=r["job_id"],
                url=r["url"],
                domain=r["domain"],
                source_tier=SourceTier(r["source_tier"]),
                is_allowlisted=bool(r["is_allowlisted"]),
                title=r["title"],
                cleaned_text=r["cleaned_text"],
                text_hash=r["text_hash"],
                published_at=r["published_at"],
                fetched_at=r["fetched_at"],
                syndication_cluster_id=r["syndication_cluster_id"],
                is_syndicated=bool(r["is_syndicated"]),
                char_count=r["char_count"],
                word_count=r["word_count"],
            )

    def get_allowlisted_sources_for_job(self, job_id: str) -> List[StoredSource]:
        """Retrieve only allowlisted sources capable of backing factual claims."""
        all_sources = self.get_sources_for_job(job_id)
        return [s for s in all_sources if s.is_allowlisted]

    # --- Fact & Conflict Operations ---

    def save_fact(self, fact: ExtractedFact) -> None:
        """Store or update an extracted fact in SQLite."""
        now = datetime.now(timezone.utc).isoformat()
        entities_json = json.dumps(fact.normalized_entities, ensure_ascii=False)
        corroborating_json = json.dumps(fact.corroborating_source_ids, ensure_ascii=False)

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO facts (
                    fact_id, job_id, claim, fact_type, normalized_entities_json,
                    normalized_value, source_id, evidence_quote, source_tier,
                    published_at, corroborating_sources_json, corroboration_status,
                    requires_attribution, attribution_outlet, is_hedged, is_temporal,
                    is_conflicted, discard_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.fact_id,
                    fact.job_id,
                    fact.claim,
                    fact.fact_type.value,
                    entities_json,
                    fact.normalized_value,
                    fact.source_id,
                    fact.evidence_quote,
                    fact.source_tier.value,
                    fact.published_at,
                    corroborating_json,
                    fact.corroboration_status.value,
                    1 if fact.requires_attribution else 0,
                    fact.attribution_outlet,
                    1 if fact.is_hedged else 0,
                    1 if fact.is_temporal else 0,
                    1 if fact.is_conflicted else 0,
                    fact.discard_reason,
                    now,
                ),
            )
        logger.debug("Stored fact %s for job %s: status=%s", fact.fact_id, fact.job_id, fact.corroboration_status.value)

    def save_facts(self, facts: List[ExtractedFact]) -> None:
        """Store multiple extracted facts in SQLite."""
        for fact in facts:
            self.save_fact(fact)

    def get_facts_for_job(self, job_id: str, include_discarded: bool = False) -> List[ExtractedFact]:
        """Retrieve extracted facts for a job."""
        with self.get_connection() as conn:
            query = """
                SELECT fact_id, job_id, claim, fact_type, normalized_entities_json,
                       normalized_value, source_id, evidence_quote, source_tier,
                       published_at, corroborating_sources_json, corroboration_status,
                       requires_attribution, attribution_outlet, is_hedged, is_temporal,
                       is_conflicted, discard_reason, created_at
                FROM facts
                WHERE job_id = ?
            """
            if not include_discarded:
                query += " AND discard_reason IS NULL"
            query += " ORDER BY corroboration_status ASC, fact_id ASC"

            rows = conn.execute(query, (job_id,)).fetchall()
            facts: List[ExtractedFact] = []
            for r in rows:
                try:
                    ents = json.loads(r["normalized_entities_json"])
                except Exception:
                    ents = []
                try:
                    corrobs = json.loads(r["corroborating_sources_json"])
                except Exception:
                    corrobs = []

                facts.append(
                    ExtractedFact(
                        fact_id=r["fact_id"],
                        job_id=r["job_id"],
                        claim=r["claim"],
                        fact_type=FactType(r["fact_type"]),
                        normalized_entities=ents,
                        normalized_value=r["normalized_value"],
                        source_id=r["source_id"],
                        evidence_quote=r["evidence_quote"],
                        source_tier=SourceTier(r["source_tier"]),
                        published_at=r["published_at"],
                        corroborating_source_ids=corrobs,
                        corroboration_status=CorroborationStatus(r["corroboration_status"]),
                        requires_attribution=bool(r["requires_attribution"]),
                        attribution_outlet=r["attribution_outlet"],
                        is_hedged=bool(r["is_hedged"]),
                        is_temporal=bool(r["is_temporal"]),
                        is_conflicted=bool(r["is_conflicted"]),
                        discard_reason=r["discard_reason"],
                    )
                )
            return facts

    def get_fact(self, fact_id: str) -> Optional[ExtractedFact]:
        """Retrieve a specific fact by ID."""
        with self.get_connection() as conn:
            r = conn.execute(
                """
                SELECT fact_id, job_id, claim, fact_type, normalized_entities_json,
                       normalized_value, source_id, evidence_quote, source_tier,
                       published_at, corroborating_sources_json, corroboration_status,
                       requires_attribution, attribution_outlet, is_hedged, is_temporal,
                       is_conflicted, discard_reason, created_at
                FROM facts
                WHERE fact_id = ?
                """,
                (fact_id,),
            ).fetchone()
            if not r:
                return None

            try:
                ents = json.loads(r["normalized_entities_json"])
            except Exception:
                ents = []
            try:
                corrobs = json.loads(r["corroborating_sources_json"])
            except Exception:
                corrobs = []

            return ExtractedFact(
                fact_id=r["fact_id"],
                job_id=r["job_id"],
                claim=r["claim"],
                fact_type=FactType(r["fact_type"]),
                normalized_entities=ents,
                normalized_value=r["normalized_value"],
                source_id=r["source_id"],
                evidence_quote=r["evidence_quote"],
                source_tier=SourceTier(r["source_tier"]),
                published_at=r["published_at"],
                corroborating_source_ids=corrobs,
                corroboration_status=CorroborationStatus(r["corroboration_status"]),
                requires_attribution=bool(r["requires_attribution"]),
                attribution_outlet=r["attribution_outlet"],
                is_hedged=bool(r["is_hedged"]),
                is_temporal=bool(r["is_temporal"]),
                is_conflicted=bool(r["is_conflicted"]),
                discard_reason=r["discard_reason"],
            )

    def save_conflict(self, conflict: ConflictRecord) -> None:
        """Store a detected fact conflict in SQLite."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO conflicts (
                    conflict_id, job_id, entity_or_topic, fact_ids_json,
                    conflicting_values_json, sources_json, resolved,
                    chosen_fact_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conflict.conflict_id,
                    conflict.job_id,
                    conflict.entity_or_topic,
                    json.dumps(conflict.fact_ids, ensure_ascii=False),
                    json.dumps(conflict.conflicting_values, ensure_ascii=False),
                    json.dumps(conflict.sources, ensure_ascii=False),
                    1 if conflict.resolved else 0,
                    conflict.chosen_fact_id,
                    conflict.created_at,
                ),
            )
        logger.info("Saved conflict %s for job %s (topic: %s)", conflict.conflict_id, conflict.job_id, conflict.entity_or_topic)

    def get_conflicts_for_job(self, job_id: str) -> List[ConflictRecord]:
        """Retrieve all conflict records for a job."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT conflict_id, job_id, entity_or_topic, fact_ids_json,
                       conflicting_values_json, sources_json, resolved,
                       chosen_fact_id, created_at
                FROM conflicts
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchall()
            conflicts: List[ConflictRecord] = []
            for r in rows:
                try:
                    fids = json.loads(r["fact_ids_json"])
                except Exception:
                    fids = []
                try:
                    vals = json.loads(r["conflicting_values_json"])
                except Exception:
                    vals = []
                try:
                    srcs = json.loads(r["sources_json"])
                except Exception:
                    srcs = []

                conflicts.append(
                    ConflictRecord(
                        conflict_id=r["conflict_id"],
                        job_id=r["job_id"],
                        entity_or_topic=r["entity_or_topic"],
                        fact_ids=fids,
                        conflicting_values=vals,
                        sources=srcs,
                        resolved=bool(r["resolved"]),
                        chosen_fact_id=r["chosen_fact_id"],
                        created_at=r["created_at"],
                    )
                )
            return conflicts

    def resolve_conflict(self, conflict_id: str, chosen_fact_id: str) -> bool:
        """Resolve a conflict record by marking chosen winning fact ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE conflicts
                SET resolved = 1, chosen_fact_id = ?
                WHERE conflict_id = ?
                """,
                (chosen_fact_id, conflict_id),
            )
            # Update the winning fact to mark is_conflicted = 0
            conn.execute(
                """
                UPDATE facts
                SET is_conflicted = 0
                WHERE fact_id = ?
                """,
                (chosen_fact_id,),
            )
            return cursor.rowcount > 0

    # --- Script Operations ---

    def save_script(self, script: GeneratedScript) -> None:
        """Store or update a generated script in SQLite."""
        beats_json = json.dumps([b.model_dump(mode="json") for b in script.beats], ensure_ascii=False)
        override_json = json.dumps(script.override_logs, ensure_ascii=False)

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scripts (
                    script_id, job_id, headline, beats_json, total_words,
                    estimated_duration_seconds, model_used, verifier_model_used,
                    lint_passed, all_supported, is_approved, approved_at,
                    override_logs_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    script.script_id,
                    script.job_id,
                    script.headline,
                    beats_json,
                    script.total_words,
                    script.estimated_duration_seconds,
                    script.model_used,
                    script.verifier_model_used,
                    1 if script.lint_passed else 0,
                    1 if script.all_supported else 0,
                    1 if script.is_approved else 0,
                    script.approved_at,
                    override_json,
                    script.created_at,
                ),
            )
        logger.debug("Saved script %s for job %s: approved=%s", script.script_id, script.job_id, script.is_approved)

    def get_script_for_job(self, job_id: str) -> Optional[GeneratedScript]:
        """Retrieve the latest script for a job."""
        with self.get_connection() as conn:
            r = conn.execute(
                """
                SELECT script_id, job_id, headline, beats_json, total_words,
                       estimated_duration_seconds, model_used, verifier_model_used,
                       lint_passed, all_supported, is_approved, approved_at,
                       override_logs_json, created_at
                FROM scripts
                WHERE job_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if not r:
                return None

            try:
                raw_beats = json.loads(r["beats_json"])
                beats = [ScriptBeat.model_validate(b) for b in raw_beats]
            except Exception:
                beats = []

            try:
                overrides = json.loads(r["override_logs_json"])
            except Exception:
                overrides = []

            return GeneratedScript(
                script_id=r["script_id"],
                job_id=r["job_id"],
                headline=r["headline"],
                beats=beats,
                total_words=r["total_words"],
                estimated_duration_seconds=r["estimated_duration_seconds"],
                model_used=r["model_used"],
                verifier_model_used=r["verifier_model_used"],
                lint_passed=bool(r["lint_passed"]),
                all_supported=bool(r["all_supported"]),
                is_approved=bool(r["is_approved"]),
                approved_at=r["approved_at"],
                override_logs=overrides,
                created_at=r["created_at"],
            )

    def approve_script(self, job_id: str) -> bool:
        """Mark the script for a job as approved by human reviewer."""
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE scripts
                SET is_approved = 1, approved_at = ?
                WHERE job_id = ?
                """,
                (now, job_id),
            )
            return cursor.rowcount > 0

    def record_beat_override(
        self,
        job_id: str,
        beat_id: int,
        action: str,
        reason: str,
        edited_text: Optional[str] = None,
    ) -> Optional[GeneratedScript]:
        """Record an explicit human override on a script beat with provenance logging."""
        script = self.get_script_for_job(job_id)
        if not script:
            return None

        now = datetime.now(timezone.utc).isoformat()
        override_entry = {
            "timestamp": now,
            "beat_id": beat_id,
            "action": action,
            "reason": reason,
            "edited_text": edited_text,
        }

        for beat in script.beats:
            if beat.beat_id == beat_id:
                beat.human_override = True
                beat.human_override_reason = reason
                if action == "approve":
                    beat.verifier_verdict = VerificationVerdict.SUPPORTED
                    beat.lint_errors = []
                elif action == "edit" and edited_text:
                    beat.spoken_text = edited_text.strip()
                    beat.word_count = len(beat.spoken_text.split())
                    beat.estimated_duration_seconds = round(beat.word_count / 2.5, 1)
                    beat.verifier_verdict = VerificationVerdict.SUPPORTED
                    beat.lint_errors = []
                elif action == "reject":
                    beat.verifier_verdict = VerificationVerdict.CONTRADICTED

        script.override_logs.append(override_entry)
        self.save_script(script)
        logger.info("Recorded human override on beat %d for job %s: action=%s, reason=%s", beat_id, job_id, action, reason)
        return self.get_script_for_job(job_id)



