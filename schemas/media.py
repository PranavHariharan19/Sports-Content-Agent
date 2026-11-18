"""Pydantic schemas for media assets, PIL stat cards, and media grounding stage."""
from enum import Enum
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class MediaType(str, Enum):
    PHOTO = "photo"
    STAT_CARD = "stat_card"
    GRAPHIC = "graphic"


class MediaAsset(BaseModel):
    """A grounded visual asset (1080x1920 9:16 vertical) mapped to a script beat."""
    asset_id: str
    job_id: str
    beat_id: int
    media_type: MediaType
    file_path: str
    source_provider: str  # wikimedia, pexels, pixabay, openverse, pil_stat_card
    attribution: Optional[str] = None
    headline: str
    caption_text: str
    start_ms: int
    end_ms: int
    duration_seconds: float
    width: int = 1080
    height: int = 1920
    query_used: Optional[str] = None

    @property
    def start_sec(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        return self.end_ms / 1000.0


class MediaStageResult(BaseModel):
    """Output summary of the media grounding pipeline stage."""
    job_id: str
    assets: List[MediaAsset]
    total_assets: int
    total_duration_seconds: float
    providers_used: List[str] = Field(default_factory=list)
