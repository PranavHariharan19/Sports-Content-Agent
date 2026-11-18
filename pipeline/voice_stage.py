"""Voice and Caption pipeline stage coordinating synthesis, timestamps, and .ass generation."""
from pathlib import Path
import re
from typing import List, Optional, Tuple

from app.logging_config import get_logger
from pipeline.caption_generator import CaptionGenerator
from pipeline.db import DatabaseManager
from pipeline.state_machine import PipelineStateMachine
from pipeline.voice_generator import DEFAULT_SPORTS_VOICE, VoiceGenerator
from schemas.common import JobStatus, StageName
from schemas.script import ScriptBeat
from schemas.voice import BeatTimestamp, VoiceStageResult, WordTimestamp

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

    @staticmethod
    def align_words_to_beats(
        beats: List[ScriptBeat],
        words: List[WordTimestamp],
    ) -> Tuple[List[WordTimestamp], List[BeatTimestamp]]:
        """Map word timestamps strictly to individual script beats based on token alignment.

        Guarantees that:
        1. Every WordTimestamp receives its respective beat_id.
        2. No word or caption chunk can cross beat boundaries.
        3. Restores ending punctuation to beat-terminating words if stripped by TTS.
        """
        if not beats or not words:
            return words, []

        def clean_tokens(s: str) -> List[str]:
            return [re.sub(r"[^\w]", "", t).lower() for t in s.split() if re.sub(r"[^\w]", "", t)]

        aligned_words: List[WordTimestamp] = []
        beat_timings: List[BeatTimestamp] = []
        word_idx = 0
        total_words = len(words)

        for b_idx, beat in enumerate(beats):
            target_tokens = clean_tokens(beat.spoken_text)
            matched_words: List[WordTimestamp] = []
            accum_tokens = 0

            # If last beat, absorb all remaining words
            if b_idx == len(beats) - 1:
                while word_idx < total_words:
                    w = words[word_idx]
                    w.beat_id = beat.beat_id
                    matched_words.append(w)
                    aligned_words.append(w)
                    word_idx += 1
            else:
                while word_idx < total_words:
                    w = words[word_idx]
                    w.beat_id = beat.beat_id
                    w_tokens = clean_tokens(w.word)
                    matched_words.append(w)
                    aligned_words.append(w)
                    word_idx += 1
                    accum_tokens += len(w_tokens)
                    if accum_tokens >= len(target_tokens):
                        break

            if matched_words:
                # Restore ending punctuation to the final word of the beat if stripped by TTS
                beat_text = beat.spoken_text.strip()
                if beat_text and beat_text[-1] in ".!?;:":
                    last_char = beat_text[-1]
                    if not matched_words[-1].word.endswith(last_char):
                        matched_words[-1].word += last_char

                b_start = matched_words[0].start_ms
                b_end = matched_words[-1].end_ms
                beat_timings.append(
                    BeatTimestamp(
                        beat_id=beat.beat_id,
                        start_ms=b_start,
                        end_ms=b_end,
                        spoken_text=beat.spoken_text,
                        word_count=len(matched_words),
                    )
                )

        return aligned_words, beat_timings

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

        audio_metadata, raw_word_timestamps = await self.voice_generator.synthesize(
            text=full_script_text,
            output_path=audio_file_path,
            job_id=job_id,
        )

        # 2. Strict Beat Alignment & Boundary Enforcement
        aligned_words, beat_timings = self.align_words_to_beats(script.beats, raw_word_timestamps)

        # 3. Synchronized ASS & SRT Captions (strictly partitioned per beat)
        subtitles_record = self.caption_generator.write_subtitles(
            words=aligned_words,
            output_dir=job_render_dir,
            job_id=job_id,
            beats=beat_timings,
        )

        result = VoiceStageResult(
            job_id=job_id,
            audio=audio_metadata,
            subtitles=subtitles_record,
            words_total=len(aligned_words),
            total_duration_seconds=audio_metadata.duration_seconds,
            beat_timings=beat_timings,
        )

        # 4. Record stage execution in DB
        self.state_machine.record_stage_execution(
            job_id=job_id,
            stage_name=StageName.VOICE,
            input_data={
                "voice": self.voice_generator.voice,
                "text_length": len(full_script_text),
                "words_count": len(full_script_text.split()),
                "beats_count": len(script.beats),
            },
            output_data=result.model_dump(mode="json"),
            success=True,
            allow_rerun=True,
        )

        logger.info(
            "Completed voice and caption stage for job %s (audio: %s, duration: %.2fs, chunks: %d, beats: %d)",
            job_id,
            audio_file_path,
            result.total_duration_seconds,
            subtitles_record.chunk_count,
            len(beat_timings),
        )

        return result
