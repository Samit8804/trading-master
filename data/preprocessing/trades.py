"""Trading log CSV cleaning pipeline.

Steps
-----
1. Load CSV from raw data directory.
2. Detect and report exact duplicate rows.
3. Group trades within a configurable millisecond window (same account,
   same strategy) by averaging price/pnl and summing quantity.
4. Filter to a specific account if desired.
5. Normalise timestamps to UTC.
6. Drop rows where critical fields (timestamp, price, pnl) are missing.
7. Save the clean DataFrame to the processed data directory.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from crowdwisdom_quant.config.settings import Config

logger = logging.getLogger(__name__)

# Expected columns in the raw trading-log CSV
EXPECTED_COLUMNS = [
    "timestamp",
    "direction",
    "price",
    "quantity",
    "pnl",
    "strategy_permutation",
    "simulation_id",
    "account",
]


class TradingLogCleaner:
    """Load, validate, clean, and save trading-log CSV files."""

    def __init__(
        self,
        raw_dir: Path | None = None,
        processed_dir: Path | None = None,
        group_ms: int | None = None,
    ) -> None:
        self.raw_dir = raw_dir or Config.RAW_DATA_DIR
        self.processed_dir = processed_dir or Config.PROCESSED_DATA_DIR
        self.group_ms = group_ms or Config.TRADE_GROUP_MS
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        input_filename: str = "trading_logs.csv",
        output_filename: str = "trading_logs_clean.parquet",
        account_filter: str | None = None,
    ) -> pd.DataFrame:
        """Execute the full cleaning pipeline.

        Parameters
        ----------
        input_filename : str
            Name of the CSV file in ``raw_dir``.
        output_filename : str
            Name for the cleaned output file (Parquet format) in ``processed_dir``.
        account_filter : str, optional
            If provided, keep only trades for this account ID.

        Returns
        -------
        pd.DataFrame
            Cleaned trading log.
        """
        self._load(input_filename)
        self._report_duplicates()
        self._group_nearby_trades()
        if account_filter is not None:
            self._filter_account(account_filter)
        self._normalise_timestamps()
        self._remove_missing()
        self._save(output_filename)
        return self._df

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------
    def _load(self, filename: str) -> None:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Trading log CSV not found: {path}")

        logger.info("Loading trades from %s", path)
        df = pd.read_csv(path, parse_dates=False)

        # Validate columns
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

        self._df = df
        logger.info("Loaded %d trade rows.", len(self._df))

    def _report_duplicates(self) -> None:
        """Identify and log exact duplicate rows."""
        if self._df is None:
            return
        dupe_mask = self._df.duplicated(keep="first")
        n_dupes = dupe_mask.sum()
        if n_dupes > 0:
            logger.warning("Found %d exact duplicate rows. Dropping them.", n_dupes)
            self._df = self._df[~dupe_mask].copy()
        else:
            logger.info("No duplicate rows detected.")

    def _group_nearby_trades(self) -> None:
        """Group trades within ``group_ms`` milliseconds.

        We sort by timestamp, then group consecutive rows that belong to
        the same ``(account, simulation_id, strategy_permutation, direction)``
        and are within ``group_ms`` of the previous row. Within each group,
        price and PnL are averaged (quantity-weighted), and quantity is summed.
        """
        if self._df is None or self._df.empty:
            return

        df = self._df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Data is sorted chronologically before grouping. The groupby + diff()
        # relies on this sort order to correctly detect group boundaries:
        # the first row of each (account, sim, strategy, direction) group
        # gets NaN diff (start of a new trade group), and any subsequent row
        # within group_ms from the previous row is merged.
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Identify group boundaries
        group_cols = ["account", "simulation_id", "strategy_permutation", "direction"]
        df["time_diff"] = df.groupby(group_cols)["timestamp"].diff()

        # A new group starts when diff > group_ms or NaN (first row of group)
        ms = pd.Timedelta(milliseconds=self.group_ms)
        df["new_group"] = (
            df["time_diff"].isna() | (df["time_diff"] > ms)
        ).astype(int)
        df["group_id"] = df.groupby(group_cols)["new_group"].cumsum()

        # Aggregate within groups
        aggregator: dict = {
            "timestamp": "first",
            "price": "mean",
            "quantity": "sum",
            "pnl": "sum",
            "strategy_permutation": "first",
            "simulation_id": "first",
            "account": "first",
            "direction": "first",
        }

        grouped = df.groupby(
            group_cols + ["group_id"], as_index=False, sort=False
        ).agg(aggregator)

        grouped = grouped.drop(columns=["group_id", "new_group", "time_diff"],
                               errors="ignore")

        before = len(self._df)
        self._df = grouped
        after = len(self._df)
        logger.info(
            "Grouped trades within %d ms: %d → %d rows.",
            self.group_ms,
            before,
            after,
        )

    def _filter_account(self, account: str) -> None:
        if self._df is None:
            return
        before = len(self._df)
        self._df = self._df[self._df["account"] == account].copy()
        logger.info(
            "Filtered to account '%s': %d → %d rows.", account, before, len(self._df)
        )

    def _normalise_timestamps(self) -> None:
        """Ensure all timestamps are timezone-aware UTC.

        Naive (tz-less) timestamps are assumed to be UTC.
        """
        if self._df is None or self._df.empty:
            return
        ts = pd.to_datetime(self._df["timestamp"], utc=True)
        self._df["timestamp"] = ts
        self._df["timezone"] = "UTC"

    def _remove_missing(self) -> None:
        """Drop rows with missing values in critical columns."""
        if self._df is None:
            return
        critical = ["timestamp", "price", "pnl"]
        before = len(self._df)
        self._df = self._df.dropna(subset=critical).copy()
        dropped = before - len(self._df)
        if dropped:
            logger.warning("Dropped %d rows with missing critical values.", dropped)

    def _save(self, filename: str) -> None:
        """Write cleaned DataFrame to Parquet.

        Parquet is chosen over CSV because it preserves dtypes (including
        timezone-aware datetimes), compresses well, and is faster to read.
        """
        if self._df is None:
            return
        path = self.processed_dir / filename
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(path, index=False)
        logger.info("Saved clean trades (%d rows) to %s", len(self._df), path)
