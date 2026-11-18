"""Pydantic schemas for voice synthesis, word-level timestamps, and ASS captions."""
from typing import List, Optional
from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    """Sub-second word boundary timestamp in milliseconds."""
    word: str
    start_ms: int = Field(description="Start time in milliseconds from start of audio")
    end_ms: int = Field(description="End time in milliseconds from start of audio")

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


class AudioMetadata(BaseModel):
    """Metadata for generated voiceover audio track."""
    audio_path: str
    duration_seconds: float
    voice_used: str
    sample_rate: int = 44100
    word_count: int
    format: str = "mp3"


class SubtitleChunk(BaseModel):
    """A 2-4 word caption chunk displayed on screen simultaneously."""
    chunk_id: int
    words: List[WordTimestamp]
    start_ms: int
    end_ms: int
    text: str


class SubtitleFileRecord(BaseModel):
    """Generated subtitle files (.ass and .srt) for a job."""
    job_id: str
    ass_path: str
    srt_path: str
    chunk_count: int
    word_count: int
    words: List[WordTimestamp] = Field(default_factory=list)


class VoiceStageResult(BaseModel):
    """Combined output of the voice synthesis and caption generation stage."""
    job_id: str
    audio: AudioMetadata
    subtitles: SubtitleFileRecord
    words_total: int
    total_duration_seconds: float
