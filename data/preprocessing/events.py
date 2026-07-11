"""Macro-event cleaning pipeline.

Steps
-----
1. Load macro events from the SQLite database via the DatabaseManager.
2. Remove rows where timestamp or event_name is missing.
3. Normalise timestamps to UTC.
4. Remove exact duplicates.
5. Save the clean DataFrame to the processed data directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from crowdwisdom_quant.config.settings import Config
from crowdwisdom_quant.data.database.manager import DatabaseManager

logger = logging.getLogger(__name__)


class MacroEventCleaner:
    """Load macro events from DB, clean, and export to Parquet."""

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        processed_dir: Path | None = None,
    ) -> None:
        self.db = db_manager or DatabaseManager()
        self.processed_dir = processed_dir or Config.PROCESSED_DATA_DIR

    def run(
        self, output_filename: str = "macro_events_clean.parquet"
    ) -> pd.DataFrame:
        """Execute the cleaning pipeline and return a clean DataFrame."""
        df = self._load_from_db()
        df = self._remove_missing(df)
        df = self._normalise_timestamps(df)
        df = self._drop_duplicates(df)
        self._save(df, output_filename)
        return df

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def _load_from_db(self) -> pd.DataFrame:
        with self.db.session() as session:
            rows = self.db.get_macro_events(session)
            if not rows:
                logger.warning("No macro events found in database.")
                return pd.DataFrame()
            records = [
                {
                    "id": r.id,
                    "timestamp": r.timestamp,
                    "event_name": r.event_name,
                    "country": r.country,
                    "forecast": r.forecast,
                    "actual": r.actual,
                    "previous": r.previous,
                    "importance": r.importance,
                    "timezone": r.timezone,
                }
                for r in rows
            ]
        df = pd.DataFrame(records)
        logger.info("Loaded %d macro events from database.", len(df))
        return df

    @staticmethod
    def _remove_missing(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=["timestamp", "event_name"]).copy()
        if before - len(df):
            logger.warning(
                "Dropped %d rows with missing timestamp/event_name.",
                before - len(df),
            )
        return df

    @staticmethod
    def _normalise_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timezone"] = "UTC"
        return df

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        subset = ["timestamp", "event_name", "country"]
        df = df.drop_duplicates(subset=subset, keep="first").copy()
        if before - len(df):
            logger.info("Removed %d duplicate macro events.", before - len(df))
        return df

    def _save(self, df: pd.DataFrame, filename: str) -> None:
        if df.empty:
            logger.warning("No macro events to save.")
            return
        path = self.processed_dir / filename
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("Saved clean macro events (%d rows) to %s", len(df), path)
