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

    def _is_stat_beat(self, beat: ScriptBeat, cited_facts: List[ExtractedFact]) -> bool:
        """Determine if a beat is best depicted as a dedicated numerical stat card."""
        # 1. Fact type check
        for f in cited_facts:
            if f.fact_type in (FactType.NUMBER, FactType.RECORD, FactType.RANKING):
                return True

        # 2. Spoken text numerical patterns (e.g., "43-game", "104th", "1.465 seconds", "12 minutes")
        has_stat_pattern = bool(
            re.search(r"\b(\d+[\.,]?\d*)\s*(?:game|match|run|second|sec|minute|min|win|point|lap|meter|yard|run|goal)s?\b", beat.spoken_text, re.IGNORECASE)
            or re.search(r"\b\d+(?:st|nd|rd|th)\b", beat.spoken_text, re.IGNORECASE)
        )
        return has_stat_pattern

    def _create_stat_card_asset(
        self,
        job_id: str,
        beat: ScriptBeat,
        cited_facts: List[ExtractedFact],
        sport,
        timing: Dict[str, float],
        output_path: Path,
    ) -> MediaAsset:
        """Generate a PIL stat card for a factual beat with numbers or records."""
        # Extract stat value and label
        stat_val = "KEY STAT"
        stat_lbl = "OFFICIAL RECORD"
        evidence_source = "VERIFIED GROUNDING"

        # Try to extract best stat from cited facts
        for f in cited_facts:
            if f.normalized_value:
                stat_val = f.normalized_value
                stat_lbl = f.claim[:36].upper()
                if f.attribution_outlet:
                    evidence_source = f"{f.attribution_outlet} // {f.source_tier.value.upper()}"
                break

        # Fallback to regex extraction from spoken text
        if stat_val == "KEY STAT":
            match = re.search(r"\b(\d+[\.,]?\d*(?:st|nd|rd|th)?)\b", beat.spoken_text)
            if match:
                stat_val = match.group(1).upper()
                stat_lbl = "MATCH DETAIL"

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
            asset_id=f"media_{job_id}_beat_{beat.beat_id}_{uuid.uuid4().hex[:6]}",
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
        """Fetch high-quality photo from open providers, falling back to PIL card if unreachable."""
        # Build candidate search queries in priority order
        queries: List[str] = []

        # 1. Named entities from cited facts
        for f in cited_facts:
            for ent in f.normalized_entities:
                if ent and len(ent) >= 3 and ent not in queries:
                    queries.append(ent)

        # 2. Extract capitalized entity names from spoken text
        named_in_text = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", beat.spoken_text)
        for n in named_in_text:
            if n not in queries and len(n) >= 4 and n.lower() not in ("what", "this", "according", "there", "when"):
                queries.append(n)

        # 3. Sport generic action fallback query
        queries.append(f"{sport.value} action celebration")

        # Execute search across queries
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
                asset_id=f"media_{job_id}_beat_{beat.beat_id}_{uuid.uuid4().hex[:6]}",
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
