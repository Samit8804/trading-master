"""Feature engineering pipeline for trading-log data merged with macro events.

All features are computed **without future leakage**:

  1. Time-based features are derived from the timestamp alone.
  2. Macro features use only the event that occurred *before* the trade
     (guaranteed by ``merge_asof(direction='backward')`` in Phase 3).
  3. Rolling / aggregate features use a **shift(1)** so the current
     trade's outcome never contributes to its own feature vector.
  4. Scaling uses ``StandardScaler`` fitted on the supplied data (intended
     to be re-fitted per training fold during walk-forward validation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from crowdwisdom_quant.config.settings import Config

logger = logging.getLogger(__name__)

# Rolling-window sizes (in number of trades) for summary statistics.
ROLLING_WINDOWS = [5, 10, 20]

# Columns to one-hot encode
CATEGORICAL_COLUMNS = [
    "direction",
    "strategy_permutation",
    "event_name",
]

# Columns that should NOT be used as features
# ``id`` and ``id_event`` are dropped in the merge layer.
NON_FEATURE_COLUMNS = [
    "macro_timestamp",
    "timezone",
    "created_at",
    "timezone_event",
    "simulation_id",
    "importance",
    "country",
    # The target itself
    "pnl",
]


class FeatureEngineer:
    """Compute, encode, and scale features from merged trade+macro data."""

    def __init__(
        self,
        processed_dir: Path | None = None,
        scaler: StandardScaler | None = None,
    ) -> None:
        self.processed_dir = processed_dir or Config.PROCESSED_DATA_DIR
        self.scaler = scaler or StandardScaler()
        self._fitted: bool = False
        self._feature_columns: List[str] = []
        self._to_scale: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        input_filename: str = "merged_data.parquet",
        output_features: str = "features.parquet",
        output_target: str = "target.parquet",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Load merged data, engineer features, and save.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix (with column names).
        y : pd.Series
            Target variable (PnL).
        """
        df = self._load(input_filename)
        df = self._add_time_features(df)
        df = self._add_macro_features(df)
        df = self._add_rolling_features(df)
        df, _ = self._add_aggregate_features(df)
        df = self._add_event_type_encoding(df)
        X, y = self._separate_target(df)
        X = self._encode_categoricals(X)
        X = self._scale(X)
        self._save(X, y, output_features, output_target)
        return X, y

    def fit_transform(
        self, df: pd.DataFrame, fit_scaler: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Full feature pipeline on an in-memory DataFrame (used in walk-forward).

        Parameters
        ----------
        df : pd.DataFrame
            Merged trade+macro data.
        fit_scaler : bool
            Whether to fit the scaler (``True`` for training fold).

        Returns
        -------
        X, y
        """
        df = self._add_time_features(df)
        df = self._add_macro_features(df)
        df = self._add_rolling_features(df)
        df, _ = self._add_aggregate_features(df)
        df = self._add_event_type_encoding(df)
        X, y = self._separate_target(df)
        X = self._encode_categoricals(X)
        X = self._scale(X, fit=fit_scaler)
        return X, y

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply already-fitted transformations (used on test folds)."""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer has not been fitted yet.")
        return self.fit_transform(df, fit_scaler=False)

    @property
    def feature_columns(self) -> List[str]:
        return self._feature_columns

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------
    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Generate calendar-based features from the trade timestamp.

        * **hour** / **minute** – intra-day patterns (e.g. higher volatility
          at market open/close).
        * **weekday** – day-of-week effects (e.g. week-end repositioning).
        * **month** / **week_number** – seasonal / monthly effects.
        * **is_weekend** – binary flag for Saturday/Sunday (many instruments
          do not trade, but crypto does; useful for filtering or weighting).
        """
        ts = df["timestamp"]
        df["hour"] = ts.dt.hour
        df["minute"] = ts.dt.minute
        df["weekday"] = ts.dt.weekday
        df["month"] = ts.dt.month
        df["week_number"] = ts.dt.isocalendar().week.astype(int)
        df["is_weekend"] = ts.dt.weekday.isin([5, 6]).astype(int)
        return df

    @staticmethod
    def _add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
        """Generate features that relate trades to macro-economic events.

        * **time_since_last_macro_event** – seconds elapsed since the most
          recent macro release. Trades right after a release may see higher
          volatility.
        * **time_until_next_macro_event** – seconds until the *next* macro
          release. This looks at the *following* event in the sorted event
          sequence.  **Important safety note**: this feature is computed
          from the macro event timeline, not from the trade timeline.
          Because ``merge_asof(direction='backward')`` guarantees that
          ``macro_timestamp <= timestamp``, the next macro event (if any)
          is always strictly after the current trade.  This feature is
          therefore safe for scheduled announcements (known in advance by
          the market).  If the event feed includes *unscheduled* surprise
          events, this feature would constitute look-ahead bias.  If you
          reuse this pattern, add an explicit guard::

              assert (df["macro_timestamp"] <= df["timestamp"]).all()
        * **macro_surprise** = actual - forecast. Positive surprises
          (actual > forecast) may indicate bullish pressure; negative
          surprises may indicate bearish pressure.
        """
        if "macro_timestamp" in df.columns:
            if not df["macro_timestamp"].isna().all():
                assert (
                    df["macro_timestamp"].dropna() <= df["timestamp"]
                ).all(), (
                    "Found macro_timestamp > timestamp — data leakage detected. "
                    "Check that merge_asof(direction='backward') is used."
                )
            df["time_since_last_macro_event"] = (
                df["timestamp"] - df["macro_timestamp"]
            ).dt.total_seconds()
        else:
            df["time_since_last_macro_event"] = np.nan

        # time_until_next_macro_event: look at the next event's timestamp
        # within the same merge_asof group.  We compute this by sorting events
        # and using shift(-1) *within* the event DataFrame, then merging again.
        # A simpler approximation: use the next macro_timestamp in sorted order.
        if "macro_timestamp" in df.columns and not df["macro_timestamp"].isna().all():
            event_times = df["macro_timestamp"].dropna().sort_values().unique()
            next_map = pd.Series(
                data=np.concatenate([event_times[1:], [pd.NaT]]),
                index=event_times,
            )
            next_ts = df["macro_timestamp"].map(next_map)
            df["time_until_next_macro_event"] = (
                next_ts - df["timestamp"]
            ).dt.total_seconds()
        else:
            df["time_until_next_macro_event"] = np.nan

        if "actual" in df.columns and "forecast" in df.columns:
            df["macro_surprise"] = df["actual"] - df["forecast"]
        else:
            df["macro_surprise"] = np.nan

        return df

    @staticmethod
    def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        """Rolling-window statistics computed over recent trades.

        CRITICAL: We use ``.shift(1)`` before computing rolling stats so that
        the current trade's own PnL never leaks into the features.

        * **rolling_avg_pnl_{w}** – mean PnL over last ``w`` trades (excluding
          current). Captures short-term momentum / mean-reversion tendency.
        * **rolling_volatility_{w}** – standard deviation of PnL. Measures
          recent risk / uncertainty.
        * **rolling_win_rate_{w}** – fraction of trades with PnL > 0 in the
          window. Recent accuracy of the strategy.
        """
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        pnl_shifted = df_sorted["pnl"].shift(1)

        for w in ROLLING_WINDOWS:
            df_sorted[f"rolling_avg_pnl_{w}"] = (
                pnl_shifted.rolling(w, min_periods=1).mean()
            )
            df_sorted[f"rolling_volatility_{w}"] = (
                pnl_shifted.rolling(w, min_periods=1).std()
            )
            win_rate = pnl_shifted.rolling(w, min_periods=1).apply(
                lambda x: (x > 0).mean()
            )
            df_sorted[f"rolling_win_rate_{w}"] = win_rate

        return df_sorted

    @staticmethod
    def _add_aggregate_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Aggregate features computed over the hour and strategy.

        * **trade_frequency** – rolling count of trades in the last hour.
          High frequency may indicate algorithmic / high-frequency strategies.
          Computed via ``numpy.searchsorted`` (O(n log n)) rather than a
          naive O(n²) loop.
        * **hourly_trade_count** – number of trades already seen in the same
          clock hour (excludes the current trade).  Uses ``cumcount() + shift``
          to avoid the forward-looking bias of ``transform('count')``.
        * **strategy_frequency** – count of trades for the same strategy
          permutation in the last 24 hours.  Uses a time-windowed rolling
          count per strategy (fold-independent, no cumulative offset issue).
        """
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        timestamps_ns = df_sorted["timestamp"].values.astype("datetime64[ns]")

        # Trade frequency: count trades in the past 60 minutes (vectorised)
        one_hour_ns = np.timedelta64(1, "h")
        left_idx = np.searchsorted(
            timestamps_ns, timestamps_ns - one_hour_ns, side="left"
        )
        df_sorted["trade_frequency"] = np.arange(len(df_sorted)) - left_idx

        # Hourly trade count: cumcount within (date, hour), lagged by 1
        df_sorted["hourly_trade_count"] = (
            df_sorted.groupby([df_sorted["timestamp"].dt.date, "hour"])
            .cumcount()
            .shift(1)
            .fillna(0)
            .astype(int)
        )

        # Strategy frequency: rolling count per strategy in the last 24 hours
        one_day_ns = np.timedelta64(24, "h")
        df_sorted["strategy_frequency"] = 0
        for _, group in df_sorted.groupby("strategy_permutation"):
            idx = group.index.values
            ts_g = timestamps_ns[idx]
            left_g = np.searchsorted(ts_g, ts_g - one_day_ns, side="left")
            df_sorted.loc[idx, "strategy_frequency"] = (
                np.arange(len(ts_g)) - left_g
            )

        return df_sorted, df_sorted["pnl"]

    @staticmethod
    def _add_event_type_encoding(df: pd.DataFrame) -> pd.DataFrame:
        """Map event names to numerical codes usable by tree models.

        This is a lightweight ordinal label rather than full one-hot (which
        is handled separately).  The mapping helps tree models split on
        event *type* (e.g. CPI vs FOMC vs GDP).
        """
        if "event_name" in df.columns:
            df["event_type_code"] = df["event_name"].astype("category").cat.codes
        else:
            df["event_type_code"] = -1
        return df

    # ------------------------------------------------------------------
    # Encoding, scaling, splitting
    # ------------------------------------------------------------------
    def _encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical columns."""
        X = pd.get_dummies(
            X,
            columns=[c for c in CATEGORICAL_COLUMNS if c in X.columns],
            drop_first=True,  # avoid dummy trap
            dummy_na=False,
        )
        return X

    def _scale(
        self, X: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Scale numerical features to zero-mean, unit-variance."""
        if fit:
            numerical = X.select_dtypes(include=[np.number]).columns.tolist()
            self._to_scale = [
                c for c in numerical
                if X[c].nunique() > 2 and X[c].std() > 0
            ]
            self._feature_columns = X.columns.tolist()
            X[self._to_scale] = self.scaler.fit_transform(
                X[self._to_scale]
            )
            self._fitted = True
        else:
            # Ensure columns match the fitted feature set
            for col in self._feature_columns:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self._feature_columns]
            # Scale only the columns seen at fit time
            common = [c for c in self._to_scale if c in X.columns]
            if common:
                X[common] = self.scaler.transform(X[common])

        return X

    @staticmethod
    def _separate_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate feature matrix from target variable."""
        y = df["pnl"].copy()
        X = df.drop(
            columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns],
            errors="ignore",
        )
        # Keep only numeric and boolean columns (XGBoost requires numeric input)
        X = X.select_dtypes(include=["number", "bool"])
        # Drop timestamp if it slipped through
        for col in ["timestamp", "account"]:
            if col in X.columns:
                X = X.drop(columns=[col])
        return X, y

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def _load(self, filename: str) -> pd.DataFrame:
        path = self.processed_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Merged data not found: {path}")
        df = pd.read_parquet(path)
        logger.info("Loaded merged data: %d rows, %d cols.", len(df), len(df.columns))
        return df

    def _save(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features_name: str,
        target_name: str,
    ) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        X.to_parquet(self.processed_dir / features_name, index=False)
        pd.DataFrame({"pnl": y}).to_parquet(
            self.processed_dir / target_name, index=False
        )
        logger.info(
            "Saved features (%s, %d cols) and target (%s).",
            features_name,
            X.shape[1],
            target_name,
        )
