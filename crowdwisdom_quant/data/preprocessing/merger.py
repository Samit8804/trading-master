"""Merge cleaned trading logs with macro events for feature engineering.

For each trade, we attach the most recent macro event that occurred
*before or at the same time* as the trade.  This is a backward-looking
merge (``method="ffill"`` after a time-ordered merge_asof) and guarantees
**no future leakage**: no trade ever sees a macro event that hasn't
happened yet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from crowdwisdom_quant.config.settings import Config

logger = logging.getLogger(__name__)


class TradeEventMerger:
    """Merge trades with the macro event that was most recent at trade time."""

    def __init__(
        self,
        processed_dir: Path | None = None,
    ) -> None:
        self.processed_dir = processed_dir or Config.PROCESSED_DATA_DIR

    def run(
        self,
        trades_filename: str = "trading_logs_clean.parquet",
        events_filename: str = "macro_events_clean.parquet",
        output_filename: str = "merged_data.parquet",
    ) -> pd.DataFrame:
        """Load cleaned trades and events, merge, and save.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame sorted by trade timestamp.
        """
        trades = self._load(trades_filename)
        events = self._load(events_filename)

        if trades.empty:
            logger.warning("No trades to merge.")
            return trades

        merged = self._merge(trades, events)
        self._save(merged, output_filename)
        return merged

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _load(self, filename: str) -> pd.DataFrame:
        path = self.processed_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Processed file not found: {path}")
        df = pd.read_parquet(path)
        logger.info("Loaded %s: %d rows.", filename, len(df))
        return df

    @staticmethod
    def _merge(trades: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            logger.warning("No macro events available — merging with empty event set.")
            # Attach null placeholder columns so downstream code still works
            for col in [
                "event_name",
                "country",
                "forecast",
                "actual",
                "previous",
                "importance",
            ]:
                trades[col] = None
            return trades

        # Sort both by timestamp for merge_asof
        trades_sorted = trades.sort_values("timestamp").reset_index(drop=True)
        events_sorted = events.sort_values("timestamp").reset_index(drop=True)

        # Rename events timestamp to avoid collision
        events_renamed = events_sorted.rename(
            columns={"timestamp": "macro_timestamp"}
        )

        # Merge_asof: for each trade, pick the nearest macro event that
        # occurred BEFORE or exactly at the trade time.
        merged = pd.merge_asof(
            trades_sorted,
            events_renamed,
            left_on="timestamp",
            right_on="macro_timestamp",
            direction="backward",  # critical: no future leakage
            suffixes=("", "_event"),
        )

        # Drop id columns — they are unique row identifiers, not features.
        # Both the trade ``id`` and the event ``id_event`` are dropped here
        # (instead of being deferred to the feature engineer) so the merging
        # layer is the single point of control for id-column removal.
        merged = merged.drop(columns=["id", "id_event"], errors="ignore")

        logger.info(
            "Merged %d trades with %d macro events.", len(trades), len(events)
        )
        return merged

    def _save(self, df: pd.DataFrame, filename: str) -> None:
        path = self.processed_dir / filename
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("Saved merged data (%d rows) to %s", len(df), path)
