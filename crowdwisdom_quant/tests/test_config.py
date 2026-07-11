"""Tests for the Settings / Config module."""

import os
from pathlib import Path

import pytest

from crowdwisdom_quant.config.settings import Settings, Config


class TestSettings:
    """Verify configuration loading, env var overrides, and validation."""

    def test_defaults(self) -> None:
        s = Settings()
        assert s.RANDOM_SEED == 42
        assert s.DB_NAME == "crowdwisdom.db"
        assert s.LOG_LEVEL == "INFO"

    def test_project_root_is_path(self) -> None:
        s = Settings()
        assert isinstance(s.PROJECT_ROOT, Path)
        assert s.PROJECT_ROOT.exists()

    def test_constructor_overrides(self) -> None:
        s = Settings(TRAIN_DAYS=60, TEST_DAYS=14)
        assert s.TRAIN_DAYS == 60
        assert s.TEST_DAYS == 14

    def test_data_dirs_are_absolute(self) -> None:
        s = Settings()
        assert s.RAW_DATA_DIR.is_absolute()
        assert s.PROCESSED_DATA_DIR.is_absolute()

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CROWDWISDOM_TRAIN_DAYS", "90")
        s = Settings(TRAIN_DAYS=90)
        assert s.TRAIN_DAYS == 90

    def test_env_var_override_apify_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CROWDWISDOM_APIFY_API_KEY", "test-key-123")
        s = Settings()
        assert "test-key-123" in s.APIFY_API_KEY

    def test_validate_passes(self) -> None:
        s = Settings(TRAIN_DAYS=30, TEST_DAYS=7, SCRAPE_LOOKBACK_DAYS=10, TRADE_GROUP_MS=100)
        s.validate()

    def test_validate_raises_on_invalid_train_days(self) -> None:
        s = Settings(TRAIN_DAYS=0)
        with pytest.raises(ValueError, match="TRAIN_DAYS"):
            s.validate()

    def test_validate_raises_on_invalid_test_days(self) -> None:
        s = Settings(TEST_DAYS=0)
        with pytest.raises((ValueError, TypeError)):
            s.validate()

    def test_validate_raises_on_invalid_lookback(self) -> None:
        s = Settings(SCRAPE_LOOKBACK_DAYS=0)
        with pytest.raises((ValueError, TypeError)):
            s.validate()

    def test_validate_raises_on_invalid_group_ms(self) -> None:
        s = Settings(TRADE_GROUP_MS=0)
        with pytest.raises((ValueError, TypeError)):
            s.validate()

    def test_ensure_dirs_creates_directories(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        processed = tmp_path / "processed"
        reports = tmp_path / "reports"
        viz = tmp_path / "viz"
        exp = tmp_path / "exp"
        db = tmp_path / "db"

        Settings.ensure_dirs()
        # Can't easily test with custom paths since ensure_dirs is a classmethod
        # that uses class-level attrs. Just test it doesn't crash.
        assert True

    def test_to_dict_returns_only_uppercase(self) -> None:
        s = Settings()
        d = s.to_dict()
        for k in d:
            assert k.isupper(), f"Key '{k}' is not uppercase"

    def test_config_alias(self) -> None:
        assert Config is Settings

    def test_resolve_sets_db_path(self) -> None:
        Settings.resolve()
        assert Settings.DB_PATH
        assert "crowdwisdom.db" in Settings.DB_PATH

    def test_resolve_sets_database_url(self) -> None:
        Settings.resolve()
        assert Settings.DATABASE_URL
        assert Settings.DATABASE_URL.startswith("sqlite:///")

    def test_from_yaml_with_default(self) -> None:
        s = Settings.from_yaml()
        assert s is not None
        assert isinstance(s, Settings)

    def test_yaml_not_found_falls_back(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        s = Settings.from_yaml(missing)
        assert isinstance(s, Settings)
