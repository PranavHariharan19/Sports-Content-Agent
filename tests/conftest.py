import os
from pathlib import Path
import tempfile
from typing import Generator
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from pipeline.db import DatabaseManager
from pipeline.state_machine import PipelineStateMachine
from schemas.config import AppConfig, LLMConfig, PipelineConfig, ProviderSettings, RootConfig


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Fixture providing a temporary SQLite database path."""
    return str(tmp_path / "test_sports_shorts.db")


@pytest.fixture
def db_manager(temp_db_path: str) -> DatabaseManager:
    """Fixture providing an initialized DatabaseManager with a clean temp DB."""
    return DatabaseManager(temp_db_path)


@pytest.fixture
def state_machine(db_manager: DatabaseManager) -> PipelineStateMachine:
    """Fixture providing a PipelineStateMachine backed by temp DB."""
    return PipelineStateMachine(db_manager)


@pytest.fixture
def sample_config(temp_db_path: str, tmp_path: Path) -> RootConfig:
    """Fixture providing a valid RootConfig pointing to temporary paths."""
    return RootConfig(
        app=AppConfig(
            name="Test Command Center",
            env="test",
            log_level="DEBUG",
            database_path=temp_db_path,
            cache_dir=str(tmp_path / "cache"),
            assets_dir=str(tmp_path / "assets"),
            renders_dir=str(tmp_path / "renders"),
        ),
        llm=LLMConfig(
            fallback_order=["mock_primary", "mock_secondary"],
            cache_enabled=True,
            timeout_seconds=5.0,
            max_retries_per_provider=2,
            providers={
                "mock_primary": ProviderSettings(
                    model="mock-p",
                    daily_request_limit=5,
                    rpm_limit=10,
                ),
                "mock_secondary": ProviderSettings(
                    model="mock-s",
                    daily_request_limit=10,
                    rpm_limit=20,
                ),
            },
        ),
        pipeline=PipelineConfig(
            max_story_age_hours=24,
            candidate_stories_count=10,
            min_verified_facts=3,
        ),
    )


@pytest.fixture
def test_client(sample_config: RootConfig) -> Generator[TestClient, None, None]:
    """Fixture providing FastAPI TestClient configured for tests."""
    app = create_app(config=sample_config)
    with TestClient(app) as client:
        yield client
