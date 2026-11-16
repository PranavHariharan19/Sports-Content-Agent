from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional

from pydantic import BaseModel

from app.logging_config import get_logger
from schemas.common import JobStatus, SourceTier, SportType, StageName
from schemas.job import JobRecord, StageRunRecord
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

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_stage_runs_job_id ON stage_runs(job_id);
CREATE INDEX IF NOT EXISTS idx_llm_cache_provider ON llm_cache(provider, model);
CREATE INDEX IF NOT EXISTS idx_sources_job_id ON sources(job_id);
CREATE INDEX IF NOT EXISTS idx_sources_syndication ON sources(syndication_cluster_id);
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

