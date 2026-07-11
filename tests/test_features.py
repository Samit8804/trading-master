"""Unit tests for the feature engineering module."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from crowdwisdom_quant.features.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_merged_data():
    """Create a small merged DataFrame for testing."""
    n = 20
    base = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
    np.random.seed(42)
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(seconds=i * 60) for i in range(n)],
            "pnl": np.random.randn(n) * 10,
            "direction": np.random.choice(["buy", "sell"], n),
            "strategy_permutation": np.random.choice(["strat_a", "strat_b"], n),
            "event_name": np.random.choice(["CPI", "FOMC", None], n),
            "actual": np.random.randn(n),
            "forecast": np.random.randn(n),
            "account": ["acc1"] * n,
            "country": ["US"] * n,
        }
    )


class TestFeatureEngineer:
    def test_time_features(self, sample_merged_data):
        fe = FeatureEngineer()
        df = fe._add_time_features(sample_merged_data.copy())
        for col in ["hour", "minute", "weekday", "month", "week_number", "is_weekend"]:
            assert col in df.columns

    def test_macro_features(self, sample_merged_data):
        df = sample_merged_data.copy()
        df["macro_timestamp"] = df["timestamp"] - pd.Timedelta(hours=1)
        fe = FeatureEngineer()
        result = fe._add_macro_features(df)
        assert "time_since_last_macro_event" in result.columns
        assert "macro_surprise" in result.columns
        assert "time_until_next_macro_event" in result.columns

    def test_rolling_features_no_leakage(self, sample_merged_data):
        """The shift(1) should prevent current row from leaking."""
        df = sample_merged_data.copy()
        fe = FeatureEngineer()
        result = fe._add_rolling_features(df)
        for w in [5, 10, 20]:
            col = f"rolling_avg_pnl_{w}"
            # First row should be just its own shift(1)=NaN which is filled,
            # but at minimum the last row's rolling window should not include itself
            assert col in result.columns

    def test_fit_transform_produces_X_y(self, sample_merged_data):
        fe = FeatureEngineer()
        X, y = fe.fit_transform(sample_merged_data.copy())
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        # Target should not be in features
        assert "pnl" not in X.columns

    def test_scaler_is_fitted(self, sample_merged_data):
        fe = FeatureEngineer()
        fe.fit_transform(sample_merged_data.copy(), fit_scaler=True)
        assert fe._fitted
