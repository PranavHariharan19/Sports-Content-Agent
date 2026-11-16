from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from schemas.common import SourceTier, SportType


class DiscoveredArticle(BaseModel):
    title: str
    url: str
    domain: str
    published_at: datetime
    summary: Optional[str] = None
    source_tier: SourceTier = SourceTier.UNKNOWN
    sport: SportType
    reddit_score: int = 0
    reddit_comments: int = 0


class ViralityBreakdown(BaseModel):
    recency_score: float = Field(default=0.0, ge=0.0, le=25.0, description="Recency score (0-25)")
    sources_score: float = Field(default=0.0, ge=0.0, le=25.0, description="Independent sources score (0-25)")
    reddit_velocity_score: float = Field(default=0.0, ge=0.0, le=25.0, description="Reddit score/comment velocity (0-25)")
    star_weight_score: float = Field(default=0.0, ge=0.0, le=15.0, description="Star entity relevance weight (0-15)")
    pageview_interest_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Wikipedia pageview velocity (0-10)")
    deterministic_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Sum of deterministic scores (0-100)")
    llm_emotion_score: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Optional LLM subjective controversy/emotion (0-10)")
    final_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Final combined score (0-100)")


class StoryCluster(BaseModel):
    cluster_id: str
    sport: SportType
    primary_headline: str
    articles: List[DiscoveredArticle]
    entities: List[str] = Field(default_factory=list)
    source_count: int
    independent_domains: List[str]
    age_hours: float
    virality: ViralityBreakdown
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DiscoveryRunResult(BaseModel):
    job_id: str
    sport: SportType
    candidates: List[StoryCluster]
    total_articles_discovered: int
    total_clusters_formed: int


class ChooseTopicRequest(BaseModel):
    cluster_id: str
