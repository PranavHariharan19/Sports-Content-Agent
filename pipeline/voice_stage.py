"""Voice and Caption pipeline stage coordinating synthesis, timestamps, and .ass generation."""
from pathlib import Path
from typing import Optional

from app.logging_config import get_logger
from pipeline.caption_generator import CaptionGenerator
from pipeline.db import DatabaseManager
from pipeline.state_machine import PipelineStateMachine
from pipeline.voice_generator import DEFAULT_SPORTS_VOICE, VoiceGenerator
from schemas.common import JobStatus, StageName
from schemas.voice import VoiceStageResult

logger = get_logger(__name__)


class VoicePipelineStage:
    """Orchestrates TTS voice synthesis and word-level highlighted ASS subtitle generation."""

    def __init__(
        self,
        db: DatabaseManager,
        voice_generator: Optional[VoiceGenerator] = None,
        caption_generator: Optional[CaptionGenerator] = None,
        state_machine: Optional[PipelineStateMachine] = None,
        renders_dir: str = "data/renders",
    ):
        self.db = db
        self.voice_generator = voice_generator or VoiceGenerator()
        self.caption_generator = caption_generator or CaptionGenerator()
        self.state_machine = state_machine or PipelineStateMachine(db)
        self.renders_dir = Path(renders_dir)

    async def process_voice_and_captions(
        self,
        job_id: str,
        voice_override: Optional[str] = None,
    ) -> VoiceStageResult:
        """Generate voiceover audio and synchronized ASS captions for an approved script."""
        job = self.db.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Allow running on SCRIPT_APPROVED or SCRIPT_DRAFTED
        if job.status not in (
            JobStatus.SCRIPT_APPROVED,
            JobStatus.SCRIPT_DRAFTED,
            JobStatus.MEDIA_READY,
            JobStatus.RENDERED,
        ):
            raise ValueError(f"Job {job_id} must be in SCRIPT_APPROVED status (current: {job.status.value})")

        script = self.db.get_script_for_job(job_id)
        if not script or not script.beats:
            raise ValueError(f"No script beats found for job {job_id}")

        # Assemble full voiceover script text from beats
        full_script_text = " ".join(b.spoken_text.strip() for b in script.beats if b.spoken_text.strip())
        if not full_script_text:
            raise ValueError(f"Script for job {job_id} has no spoken text")

        job_render_dir = self.renders_dir / job_id
        job_render_dir.mkdir(parents=True, exist_ok=True)
        audio_file_path = job_render_dir / "voice.mp3"

        # 1. Voice Synthesis & Word Timestamps
        if voice_override:
            self.voice_generator.voice = voice_override

        audio_metadata, word_timestamps = await self.voice_generator.synthesize(
            text=full_script_text,
            output_path=audio_file_path,
            job_id=job_id,
        )

        # 2. Synchronized ASS & SRT Captions
        subtitles_record = self.caption_generator.write_subtitles(
            words=word_timestamps,
            output_dir=job_render_dir,
            job_id=job_id,
        )

        result = VoiceStageResult(
            job_id=job_id,
            audio=audio_metadata,
            subtitles=subtitles_record,
            words_total=len(word_timestamps),
            total_duration_seconds=audio_metadata.duration_seconds,
        )

        # 3. Record stage execution in DB
        self.state_machine.record_stage_execution(
            job_id=job_id,
            stage_name=StageName.VOICE,
            input_data={
                "voice": self.voice_generator.voice,
                "text_length": len(full_script_text),
                "words_count": len(full_script_text.split()),
            },
            output_data=result.model_dump(mode="json"),
            success=True,
            allow_rerun=True,
        )

        logger.info(
            "Completed voice and caption stage for job %s (audio: %s, duration: %.2fs, chunks: %d)",
            job_id,
            audio_file_path,
            result.total_duration_seconds,
            subtitles_record.chunk_count,
        )

        return result
