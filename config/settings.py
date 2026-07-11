"""Configuration management for CrowdWisdomQuant.

All paths, database settings, API keys, and model hyperparameters
are managed centrally.  Supports:

* Environment-variable overrides (prefix ``CROWDWISDOM_``)
* YAML config files (``config/default.yaml``)
* Runtime overrides via the ``Settings`` constructor

Usage::

    from crowdwisdom_quant.config.settings import Settings

    settings = Settings()                     # defaults
    settings = Settings.from_yaml("my.yaml")  # from file
    db_url = settings.database_url            # access as attribute
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "default.yaml"


class Settings:
    """Application settings with YAML + env-var override support.

    Keeps backward compatibility with the original ``Config`` class
    while adding YAML-based configuration, environment variable
    overrides, and validation.
    """

    # ------------------------------------------------------------------
    # Class-level defaults (overridden by YAML, then by env vars)
    # ------------------------------------------------------------------
    PROJECT_ROOT: Path = _PROJECT_ROOT

    # Data directories
    RAW_DATA_DIR: Path = _PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: Path = _PROJECT_ROOT / "data" / "processed"
    DATABASE_DIR: Path = _PROJECT_ROOT / "data" / "database"
    REPORTS_DIR: Path = _PROJECT_ROOT / "reports"
    VISUALIZATION_DIR: Path = _PROJECT_ROOT / "visualization"
    EXPERIMENTS_DIR: Path = _PROJECT_ROOT / "experiments"
    NOTEBOOKS_DIR: Path = _PROJECT_ROOT / "notebooks"

    # Database
    DB_NAME: str = "crowdwisdom.db"
    DB_PATH: str = ""
    DATABASE_URL: str = ""

    # Scraper
    APIFY_API_KEY: str = ""
    APIFY_MACRO_DATASET_ID: str = "macro-economic-calendar"
    SCRAPE_LOOKBACK_DAYS: int = 180

    # Preprocessing
    TRADE_GROUP_MS: int = 500

    # Walk-forward validation
    TRAIN_DAYS: int = 30
    TEST_DAYS: int = 7
    EXPANDING_WINDOW: bool = False

    # XGBoost defaults
    MODEL_DEFAULTS: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 1,
    }
    HYPERPARAM_GRID: Dict[str, list] = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    # Target
    TARGET: str = "pnl"

    # Reproducibility
    RANDOM_SEED: int = 42

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "structured"
    LOG_FILE: str = "logs/crowdwisdom.log"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "30 days"

    # Rolling windows & categorical columns
    ROLLING_WINDOWS: List[int] = [5, 10, 20]
    CATEGORICAL_COLUMNS: List[str] = [
        "direction", "strategy_permutation", "event_name",
    ]

    def __init__(self, **overrides: Any) -> None:
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Apply remaining env var overrides
        self._apply_env_overrides()

    @classmethod
    def resolve(cls) -> None:
        """Resolve computed settings (database URL, API key, etc.).

        Called automatically on first access to ``DATABASE_URL`` or ``DB_PATH``
        via the module-level ``db_path`` / ``database_url`` properties.
        """
        if not cls.DB_PATH:
            cls.DB_PATH = os.getenv(
                "CROWDWISDOM_DB_PATH",
                str(cls.DATABASE_DIR / cls.DB_NAME),
            )
        if not cls.DATABASE_URL:
            cls.DATABASE_URL = os.getenv(
                "CROWDWISDOM_DATABASE_URL",
                f"sqlite:///{cls.DB_PATH}",
            )
        if not cls.APIFY_API_KEY:
            cls.APIFY_API_KEY = os.getenv("APIFY_API_KEY", "")

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "Settings":
        """Load settings from a YAML file.

        Falls back to ``config/default.yaml`` if *path* is ``None``.
        Values are then overridable via environment variables.
        """
        path = path or _DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Flatten nested YAML into dot-separated uppercase keys
        flat = cls._flatten(data)
        return cls(**flat)

    @staticmethod
    def _flatten(data: dict, prefix: str = "") -> dict:
        """Convert nested YAML dict to flat dict with uppercase keys."""
        result = {}
        for key, value in data.items():
            upper_key = key.upper()
            full = f"{prefix}_{upper_key}" if prefix else upper_key
            if isinstance(value, dict):
                result.update(Settings._flatten(value, full))
            else:
                result[full] = value
        return result

    @staticmethod
    def _apply_env_overrides() -> None:
        """Apply ``CROWDWISDOM_*`` environment variable overrides."""
        prefix = "CROWDWISDOM_"
        for env_key, env_val in os.environ.items():
            if env_key.startswith(prefix) and env_val:
                attr = env_key[len(prefix):]
                if hasattr(Settings, attr):
                    setattr(Settings, attr, env_val)

    def validate(self) -> None:
        """Validate configuration, raising ``ValueError`` on invalid settings."""
        train_days = int(self.TRAIN_DAYS)
        test_days = int(self.TEST_DAYS)
        lookback = int(self.SCRAPE_LOOKBACK_DAYS)
        group_ms = int(self.TRADE_GROUP_MS)

        if train_days <= 0:
            raise ValueError(f"TRAIN_DAYS must be > 0, got {self.TRAIN_DAYS}")
        if test_days <= 0:
            raise ValueError(f"TEST_DAYS must be > 0, got {self.TEST_DAYS}")
        if lookback <= 0:
            raise ValueError(
                f"SCRAPE_LOOKBACK_DAYS must be > 0, got {self.SCRAPE_LOOKBACK_DAYS}"
            )
        if group_ms <= 0:
            raise ValueError(f"TRADE_GROUP_MS must be > 0, got {self.TRADE_GROUP_MS}")

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create all required directories if they don't already exist."""
        for dir_ in [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.DATABASE_DIR,
            cls.REPORTS_DIR,
            cls.VISUALIZATION_DIR,
            cls.EXPERIMENTS_DIR,
        ]:
            dir_.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Export all settings as a flat dict (for experiment logging)."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__class__.__dict__.items()
            if k.isupper() and not k.startswith("_")
        }


# Keep the original ``Config`` name for backward compatibility
Config = Settings

# Resolve computed settings at import time (backward compat with old Config)
Settings.resolve()
