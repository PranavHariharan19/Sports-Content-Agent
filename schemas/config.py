import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import yaml
from pydantic import BaseModel, Field

# Automatically load environment variables from .env if present
load_dotenv()


class AppConfig(BaseModel):
    name: str = "Sports Shorts Command Center"
    env: str = "development"
    log_level: str = "INFO"
    database_path: str = "data/sports_shorts.db"
    cache_dir: str = "data/cache"
    assets_dir: str = "data/assets"
    renders_dir: str = "data/renders"


class ProviderSettings(BaseModel):
    model: str
    daily_request_limit: int = 1000
    rpm_limit: int = 15
    base_url: Optional[str] = None


class LLMConfig(BaseModel):
    fallback_order: List[str] = Field(default_factory=lambda: ["gemini", "groq", "openrouter", "ollama"])
    cache_enabled: bool = True
    timeout_seconds: float = 30.0
    max_retries_per_provider: int = 2
    providers: Dict[str, ProviderSettings] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    max_story_age_hours: int = 24
    candidate_stories_count: int = 12
    min_verified_facts: int = 3
    target_duration_seconds: int = 35
    min_words: int = 80
    max_words: int = 120
    fuzzy_evidence_threshold: float = 0.85
    max_script_retries: int = 3


class RootConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RootConfig":
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Allow environment variable overrides for database_path, log_level, etc.
        if "DATABASE_PATH" in os.environ:
            data.setdefault("app", {})["database_path"] = os.environ["DATABASE_PATH"]
        if "LOG_LEVEL" in os.environ:
            data.setdefault("app", {})["log_level"] = os.environ["LOG_LEVEL"]
        if "APP_ENV" in os.environ:
            data.setdefault("app", {})["env"] = os.environ["APP_ENV"]

        return cls.model_validate(data)


def load_config(config_path: Optional[str] = None) -> RootConfig:
    """Load configuration from the provided path or default location."""
    path = config_path or os.environ.get("CONFIG_PATH", "config/config.yaml")
    return RootConfig.from_yaml(path)
