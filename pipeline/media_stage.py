"""Media Grounding Pipeline Stage (Section 9).

Coordinates visual asset retrieval across open providers (Wikimedia, Pexels, Pixabay)
and bespoke 1080x1920 PIL stat cards/graphics grounded in verified facts.
"""
import json
from pathlib import Path
import re
from typing import Dict, List, Optional
import uuid

from app.logging_config import get_logger
from pipeline.db import DatabaseManager
from pipeline.media.card_generator import MediaCardGenerator
from pipeline.media.fetcher import MediaFetcher
from pipeline.state_machine import PipelineStateMachine
from schemas.common import FactType, JobStatus, StageName
from schemas.facts import ExtractedFact
from schemas.media import MediaAsset, MediaStageResult, MediaType
from schemas.script import GeneratedScript, ScriptBeat
from schemas.voice import VoiceStageResult

logger = get_logger(__name__)


STOP_WORDS = {
    "did", "what", "when", "where", "why", "how", "who", "which",
    "is", "are", "was", "were", "can", "could", "would", "should",
    "the", "this", "that", "these", "those", "there", "here",
    "according", "despite", "after", "before", "while", "with", "without",
    "from", "into", "over", "under", "first", "second", "third", "fourth",
    "last", "next", "every", "some", "many", "more", "most",
    "just", "also", "even", "only", "in", "on", "at", "to", "for", "of", "by", "an", "a", "and", "or", "but", "not", "it", "its", "as"
}


class MediaPipelineStage:
    """Orchestrates media retrieval and PIL graphic generation for each script beat."""

    def __init__(
        self,
        db: DatabaseManager,
        state_machine: Optional[PipelineStateMachine] = None,
        fetcher: Optional[MediaFetcher] = None,
        card_generator: Optional[MediaCardGenerator] = None,
        assets_dir: str = "data/assets",
    ):
        self.db = db
        self.state_machine = state_machine or PipelineStateMachine(db)
        self.fetcher = fetcher or MediaFetcher()
        self.card_generator = card_generator or MediaCardGenerator()
        self.assets_dir = Path(assets_dir)

    async def process_media_for_job(self, job_id: str) -> MediaStageResult:
        """Ground and generate 1080x1920 vertical visual assets for all beats in the approved script."""
        job = self.db.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status not in (
            JobStatus.SCRIPT_APPROVED,
            JobStatus.MEDIA_READY,
            JobStatus.RENDERED,
        ):
            raise ValueError(f"Job {job_id} must be in SCRIPT_APPROVED status (current: {job.status.value})")

        script = self.db.get_script_for_job(job_id)
        if not script or not script.beats:
            raise ValueError(f"No script beats found for job {job_id}")

        facts = self.db.get_facts_for_job(job_id)
        facts_by_id: Dict[str, ExtractedFact] = {f.fact_id: f for f in facts}

        # 1. Retrieve beat timings from Voice stage run if available
        beat_timings = self._get_beat_timings(job_id, script)

        job_assets_dir = self.assets_dir / job_id
        job_assets_dir.mkdir(parents=True, exist_ok=True)

        assets: List[MediaAsset] = []
        providers_used = set()

        # 2. Process visual asset for each beat
        for beat in script.beats:
            timing = beat_timings.get(
                beat.beat_id,
                {
                    "start_ms": int((beat.beat_id - 1) * 7000),
                    "end_ms": int(beat.beat_id * 7000),
                    "duration_seconds": beat.estimated_duration_seconds or 7.0,
                },
            )

            asset_file = job_assets_dir / f"beat_{beat.beat_id}.jpg"
            cited_facts = [facts_by_id[fid] for fid in beat.cited_fact_ids if fid in facts_by_id]

            # Decide asset type: Stat Card vs Photo
            is_stat_beat = self._is_stat_beat(beat, cited_facts)

            if is_stat_beat:
                # Generate PIL Stat Card
                asset = self._create_stat_card_asset(
                    job_id=job_id,
                    beat=beat,
                    cited_facts=cited_facts,
                    sport=job.sport,
                    timing=timing,
                    output_path=asset_file,
                )
            else:
                # Search and retrieve photo from Wikimedia / Pexels / Pixabay
                asset = await self._fetch_or_fallback_photo(
                    job_id=job_id,
                    beat=beat,
                    cited_facts=cited_facts,
                    sport=job.sport,
                    timing=timing,
                    output_path=asset_file,
                )

            assets.append(asset)
            providers_used.add(asset.source_provider)

        # 3. Save media assets to database
        self.db.save_media_assets(assets)

        total_duration = assets[-1].end_sec if assets else 0.0
        result = MediaStageResult(
            job_id=job_id,
            assets=assets,
            total_assets=len(assets),
            total_duration_seconds=round(total_duration, 2),
            providers_used=sorted(list(providers_used)),
        )

        # 4. Advance pipeline state to MEDIA_READY
        self.state_machine.record_stage_execution(
            job_id=job_id,
            stage_name=StageName.MEDIA,
            input_data={
                "beats_count": len(script.beats),
                "sport": job.sport.value,
            },
            output_data=result.model_dump(mode="json"),
            success=True,
            allow_rerun=True,
        )

        logger.info(
            "Completed media grounding stage for job %s: %d assets generated (providers: %s)",
            job_id,
            len(assets),
            ", ".join(providers_used),
        )

        return result

    def _get_beat_timings(self, job_id: str, script: GeneratedScript) -> Dict[int, Dict[str, float]]:
        """Extract sub-second timing bounds per beat from the latest voice stage run."""
        timings: Dict[int, Dict[str, float]] = {}
        latest_run = self.db.get_latest_stage_run(job_id=job_id, stage_name=StageName.VOICE)

        if latest_run and latest_run.status == "COMPLETED":
            try:
                data = json.loads(latest_run.output_json)
                voice_res = VoiceStageResult(**data)
                for bt in voice_res.beat_timings:
                    timings[bt.beat_id] = {
                        "start_ms": bt.start_ms,
                        "end_ms": bt.end_ms,
                        "duration_seconds": bt.duration_ms / 1000.0,
                    }
            except Exception as e:
                logger.debug("Could not parse beat timings from voice stage: %s", str(e))

        # Fallback estimation if voice stage hasn't run yet
        if not timings:
            accum_ms = 0
            for b in script.beats:
                dur_ms = int((b.estimated_duration_seconds or 6.0) * 1000)
                timings[b.beat_id] = {
                    "start_ms": accum_ms,
                    "end_ms": accum_ms + dur_ms,
                    "duration_seconds": dur_ms / 1000.0,
                }
                accum_ms += dur_ms

        return timings

    def _clean_entity(self, text: str) -> Optional[str]:
        """Strip possessives and boundary stop words from an entity name."""
        t = re.sub(r"['’]s\b", "", text).strip()
        tokens = t.split()
        while tokens and tokens[0].lower() in STOP_WORDS:
            tokens.pop(0)
        while tokens and tokens[-1].lower() in STOP_WORDS:
            tokens.pop()
        cleaned = " ".join(tokens).strip()
        if len(cleaned) >= 3 and cleaned.lower() not in STOP_WORDS and not cleaned.isdigit():
            return cleaned
        return None

    def _is_person_name(self, name: str) -> bool:
        """Heuristic to determine if a multi-word entity is likely a person."""
        words = name.split()
        if len(words) < 2 or len(words) > 3:
            return False
        lower = name.lower()
        excluded = {
            "nations league", "uefa nations", "world cup", "premier league",
            "champions league", "grand prix", "first half", "second half",
            "extra time", "injury time", "penalty spot", "penalty kick",
            "red card", "yellow card", "pit stop", "safety car",
            "wanda metropolitano", "silverstone circuit"
        }
        return not any(ex in lower for ex in excluded)

    def _extract_people(self, beat: ScriptBeat, cited_facts: List[ExtractedFact]) -> List[str]:
        """Extract athlete and manager names from cited facts and spoken text."""
        found: List[str] = []
        for f in cited_facts:
            for ent in f.normalized_entities:
                cleaned = self._clean_entity(ent)
                if cleaned and self._is_person_name(cleaned):
                    if cleaned not in found:
                        found.append(cleaned)

        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", beat.spoken_text)
        for c in candidates:
            cleaned = self._clean_entity(c)
            if cleaned and self._is_person_name(cleaned):
                if cleaned not in found:
                    found.append(cleaned)
        return found

    def _has_named_person(self, beat: ScriptBeat, cited_facts: List[ExtractedFact]) -> bool:
        """Check if the beat explicitly features a named athlete or coach."""
        return len(self._extract_people(beat, cited_facts)) > 0

    def _build_queries_for_beat(
        self, beat: ScriptBeat, cited_facts: List[ExtractedFact], sport
    ) -> List[str]:
        """Construct targeted, sports-grounded search queries in strict priority order."""
        queries: List[str] = []

        # 1. Athlete or manager names (highest visual relevance for sports fans)
        people = self._extract_people(beat, cited_facts)
        for p in people:
            q_sport = f"{p} {sport.value}"
            if q_sport not in queries:
                queries.append(q_sport)
            if p not in queries:
                queries.append(p)

        # 2. Venues / Stadiums
        stadium_match = re.search(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Stadium|Metropolitano|Arena|Circuit|Track|Ground|Park))\b",
            beat.spoken_text,
        )
        if stadium_match:
            venue = self._clean_entity(stadium_match.group(1))
            if venue:
                queries.append(f"{venue} stadium")
                queries.append(venue)

        # 3. Teams / Nations with sports context
        teams: List[str] = []
        for f in cited_facts:
            for ent in f.normalized_entities:
                cleaned = self._clean_entity(ent)
                if cleaned and not self._is_person_name(cleaned):
                    if cleaned not in teams and cleaned not in queries:
                        teams.append(cleaned)

        for word in re.findall(r"\b[A-Z][a-z]+\b", beat.spoken_text):
            cleaned = self._clean_entity(word)
            if cleaned and len(cleaned) >= 4 and not self._is_person_name(cleaned) and cleaned not in teams:
                teams.append(cleaned)

        for t in teams:
            if beat.beat_type.value in ("hook", "resolution") or "win" in beat.spoken_text.lower() or "stun" in beat.spoken_text.lower():
                queries.append(f"{t} national {sport.value} team celebration")
                queries.append(f"{t} {sport.value} celebration")
            queries.append(f"{t} national {sport.value} team")
            queries.append(f"{t} {sport.value}")

        # 4. Contextual action fallbacks
        if beat.beat_type.value == "hook" or "goal" in beat.spoken_text.lower() or "stun" in beat.spoken_text.lower():
            queries.append(f"{sport.value} goal celebration")
            queries.append(f"{sport.value} match action")
        elif "crowd" in beat.spoken_text.lower() or "stadium" in beat.spoken_text.lower() or beat.beat_type.value == "closing":
            queries.append(f"stadium {sport.value} crowd")
            queries.append(f"{sport.value} stadium atmosphere")
        else:
            queries.append(f"{sport.value} match action")
            queries.append(f"{sport.value} action celebration")

        return queries

    def _is_stat_beat(self, beat: ScriptBeat, cited_facts: List[ExtractedFact]) -> bool:
        """Determine if a beat is best depicted as a dedicated numerical stat card.
        
        Rules:
        - If the beat names an athlete or manager, always prefer their photo.
        - Never generate stat cards for match minutes (e.g. 65 minutes).
        - Only generate for genuine milestones / records (e.g. 43-game unbeaten run).
        """
        if self._has_named_person(beat, cited_facts):
            return False

        has_record_fact = any(
            f.fact_type == FactType.RECORD
            and f.normalized_value
            and any(c.isdigit() for c in f.normalized_value)
            for f in cited_facts
        )
        if has_record_fact:
            return True

        has_streak_pattern = bool(
            re.search(r"\b(\d+)\s*-(?:game|match|race|run|win|goal)\b", beat.spoken_text, re.IGNORECASE)
            or re.search(r"\b(\d+)\s+(?:games|matches|races|runs|wins)\s+unbeaten\b", beat.spoken_text, re.IGNORECASE)
            or re.search(r"\b\d+(?:st|nd|rd|th)\s+(?:win|title|pole|goal|century)\b", beat.spoken_text, re.IGNORECASE)
        )
        return has_streak_pattern

    def _create_stat_card_asset(
        self,
        job_id: str,
        beat: ScriptBeat,
        cited_facts: List[ExtractedFact],
        sport,
        timing: Dict[str, float],
        output_path: Path,
    ) -> MediaAsset:
        """Generate a PIL stat card for an official milestone or numerical streak."""
        stat_val = None
        stat_lbl = "OFFICIAL RECORD"
        evidence_source = "VERIFIED GROUNDING"

        # 1. Search for RECORD facts with digits (skip quotes and sentiment words)
        for f in cited_facts:
            if f.fact_type == FactType.RECORD and f.normalized_value and any(c.isdigit() for c in f.normalized_value):
                stat_val = f"{f.normalized_value} GAMES" if "game" in f.claim.lower() or "unbeaten" in f.claim.lower() else f.normalized_value
                stat_lbl = "UNBEATEN STREAK" if "unbeaten" in f.claim.lower() else (f.claim[:36].upper() if f.claim else "OFFICIAL RECORD")
                if f.attribution_outlet:
                    evidence_source = f"{f.attribution_outlet} // {f.source_tier.value.upper()}"
                break

        # 2. Search for explicit streak pattern in spoken text
        if not stat_val:
            m = re.search(r"\b(\d+)\s*-(?:game|match|race|run|win|goal)s?\b", beat.spoken_text, re.IGNORECASE)
            if m:
                stat_val = f"{m.group(1)} GAMES"
                stat_lbl = "UNBEATEN STREAK"
            else:
                m2 = re.search(r"\b(\d+(?:st|nd|rd|th))\b", beat.spoken_text, re.IGNORECASE)
                if m2:
                    stat_val = m2.group(1).upper()
                    stat_lbl = "CAREER MILESTONE"

        if not stat_val:
            stat_val = "UNBEATEN"
            stat_lbl = "HISTORIC STREAK"

        self.card_generator.generate_stat_card(
            headline=beat.spoken_text,
            stat_value=stat_val,
            stat_label=stat_lbl,
            detail_text=beat.spoken_text,
            sport=sport,
            output_path=output_path,
            evidence_source=evidence_source,
        )

        return MediaAsset(
            asset_id=f"media_{job_id}_beat_{beat.beat_id}",
            job_id=job_id,
            beat_id=beat.beat_id,
            media_type=MediaType.STAT_CARD,
            file_path=str(output_path),
            source_provider="pil_stat_card",
            attribution=evidence_source,
            headline=stat_val,
            caption_text=beat.spoken_text,
            start_ms=int(timing["start_ms"]),
            end_ms=int(timing["end_ms"]),
            duration_seconds=round(timing["duration_seconds"], 2),
            query_used=stat_val,
        )

    async def _fetch_or_fallback_photo(
        self,
        job_id: str,
        beat: ScriptBeat,
        cited_facts: List[ExtractedFact],
        sport,
        timing: Dict[str, float],
        output_path: Path,
    ) -> MediaAsset:
        """Fetch high-quality photo from open providers using targeted entity queries."""
        queries = self._build_queries_for_beat(beat, cited_facts, sport)

        fetch_result = None
        used_query = None

        for q in queries:
            fetch_result = await self.fetcher.search_and_download(
                query=q,
                output_path=output_path,
                sport=sport,
            )
            if fetch_result:
                used_query = q
                break

        if fetch_result:
            provider, attribution = fetch_result
            return MediaAsset(
                asset_id=f"media_{job_id}_beat_{beat.beat_id}",
                job_id=job_id,
                beat_id=beat.beat_id,
                media_type=MediaType.PHOTO,
                file_path=str(output_path),
                source_provider=provider,
                attribution=attribution,
                headline=used_query.title() if used_query else beat.spoken_text[:30],
                caption_text=beat.spoken_text,
                start_ms=int(timing["start_ms"]),
                end_ms=int(timing["end_ms"]),
                duration_seconds=round(timing["duration_seconds"], 2),
                query_used=used_query,
            )

        # Graceful fallback: generate a styled PIL graphic card if network providers yield no image
        logger.info("No network image found for beat %d; falling back to PIL card", beat.beat_id)
        return self._create_stat_card_asset(
            job_id=job_id,
            beat=beat,
            cited_facts=cited_facts,
            sport=sport,
            timing=timing,
            output_path=output_path,
        )

    async def replace_beat_visual(
        self,
        job_id: str,
        beat_id: int,
        query: Optional[str] = None,
        provider: Optional[str] = "all",
        media_type: Optional[str] = "photo",
        stat_value: Optional[str] = None,
        stat_label: Optional[str] = None,
        detail_text: Optional[str] = None,
    ) -> MediaAsset:
        """Manually replace or regenerate the visual asset for a single beat."""
        job = self.db.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        script = self.db.get_script_for_job(job_id)
        if not script or not script.beats:
            raise ValueError(f"No script beats found for job {job_id}")

        target_beat = next((b for b in script.beats if b.beat_id == beat_id), None)
        if not target_beat:
            raise ValueError(f"Beat {beat_id} not found in script for job {job_id}")

        existing_asset = self.db.get_media_asset_for_beat(job_id, beat_id)
        beat_timings = self._get_beat_timings(job_id, script)
        timing = beat_timings.get(
            beat_id,
            {
                "start_ms": existing_asset.start_ms if existing_asset else 0,
                "end_ms": existing_asset.end_ms if existing_asset else 5000,
                "duration_seconds": existing_asset.duration_seconds if existing_asset else 5.0,
            },
        )

        job_assets_dir = self.assets_dir / job_id
        job_assets_dir.mkdir(parents=True, exist_ok=True)
        output_path = job_assets_dir / f"beat_{beat_id}.jpg"

        facts = self.db.get_facts_for_job(job_id)
        facts_by_id = {f.fact_id: f for f in facts}
        cited_facts = [facts_by_id[fid] for fid in target_beat.cited_fact_ids if fid in facts_by_id]

        if media_type == "stat_card":
            val = stat_value.strip() if stat_value and stat_value.strip() else "KEY STAT"
            lbl = stat_label.strip() if stat_label and stat_label.strip() else "OFFICIAL RECORD"
            dtl = detail_text.strip() if detail_text and detail_text.strip() else target_beat.spoken_text
            self.card_generator.generate_stat_card(
                headline=target_beat.spoken_text,
                stat_value=val,
                stat_label=lbl,
                detail_text=dtl,
                sport=job.sport,
                output_path=output_path,
                evidence_source="USER CUSTOMIZED",
            )
            updated_asset = MediaAsset(
                asset_id=f"media_{job_id}_beat_{beat_id}",
                job_id=job_id,
                beat_id=beat_id,
                media_type=MediaType.STAT_CARD,
                file_path=str(output_path),
                source_provider="pil_stat_card",
                attribution="User Custom Stat Card",
                headline=val,
                caption_text=target_beat.spoken_text,
                start_ms=int(timing["start_ms"]),
                end_ms=int(timing["end_ms"]),
                duration_seconds=round(timing["duration_seconds"], 2),
                query_used=val,
            )
        else:
            q = query.strip() if query and query.strip() else self._build_queries_for_beat(target_beat, cited_facts, job.sport)[0]
            pref = None if provider == "all" else provider
            fetch_res = await self.fetcher.search_and_download(
                query=q,
                output_path=output_path,
                sport=job.sport,
                prefer_provider=pref,
            )
            if not fetch_res:
                raise ValueError(f"Could not find any suitable images for query '{q}' (provider: {provider})")

            prov, attr = fetch_res
            updated_asset = MediaAsset(
                asset_id=f"media_{job_id}_beat_{beat_id}",
                job_id=job_id,
                beat_id=beat_id,
                media_type=MediaType.PHOTO,
                file_path=str(output_path),
                source_provider=prov,
                attribution=attr,
                headline=q.title(),
                caption_text=target_beat.spoken_text,
                start_ms=int(timing["start_ms"]),
                end_ms=int(timing["end_ms"]),
                duration_seconds=round(timing["duration_seconds"], 2),
                query_used=q,
            )

        self.db.save_media_assets([updated_asset])
        return updated_asset
