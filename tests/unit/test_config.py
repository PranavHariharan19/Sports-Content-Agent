import os
from pathlib import Path
import pytest
from schemas.config import RootConfig, load_config


def test_load_default_config():
    """Verify loading default config/config.yaml."""
    cfg = load_config("config/config.yaml")
    assert cfg.app.name == "Sports Shorts Command Center"
    assert "gemini" in cfg.llm.fallback_order
    assert "groq" in cfg.llm.fallback_order
    assert cfg.llm.providers["gemini"].model == "gemini-3.5-flash-lite"
    assert cfg.pipeline.min_verified_facts >= 3


def test_missing_config_raises_file_not_found(tmp_path: Path):
    """Verify FileNotFoundError is raised for a nonexistent config path."""
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nonexistent.yaml"))


def test_env_var_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify environment variables override config values."""
    custom_db = str(tmp_path / "custom.db")
    monkeypatch.setenv("DATABASE_PATH", custom_db)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("APP_ENV", "testing")

    cfg = load_config("config/config.yaml")
    assert cfg.app.database_path == custom_db
    assert cfg.app.log_level == "DEBUG"
    assert cfg.app.env == "testing"
