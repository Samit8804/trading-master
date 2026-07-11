"""Unit tests for the preprocessing / cleaning modules."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crowdwisdom_quant.preprocessing.clean_trades import TradingLogCleaner


@pytest.fixture
def temp_dirs():
    """Provide temporary raw and processed directories."""
    raw = Path(tempfile.mkdtemp())
    processed = Path(tempfile.mkdtemp())
    yield raw, processed
    import shutil
    shutil.rmtree(raw)
    shutil.rmtree(processed)


def _write_csv(raw_dir: Path, data: pd.DataFrame, name: str = "trading_logs.csv"):
    path = raw_dir / name
    data.to_csv(path, index=False)
    return path


class TestTradingLogCleaner:
    def test_basic_cleaning(self, temp_dirs):
        raw_dir, processed_dir = temp_dirs
        df_in = pd.DataFrame(
            {
                "timestamp": ["2024-01-15 10:00:00+00:00", "2024-01-15 10:00:05+00:00"],
                "direction": ["buy", "sell"],
                "price": [100.0, 101.0],
                "quantity": [10, 20],
                "pnl": [5.0, -3.0],
                "strategy_permutation": ["a", "a"],
                "simulation_id": ["s1", "s1"],
                "account": ["acc1", "acc1"],
            }
        )
        _write_csv(raw_dir, df_in)

        cleaner = TradingLogCleaner(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            group_ms=500,
        )
        result = cleaner.run(account_filter="acc1")
        assert len(result) == 2
        assert "timezone" in result.columns

    def test_duplicate_removal(self, temp_dirs):
        raw_dir, processed_dir = temp_dirs
        row = {
            "timestamp": "2024-01-15 10:00:00+00:00",
            "direction": "buy",
            "price": 100.0,
            "quantity": 10,
            "pnl": 5.0,
            "strategy_permutation": "a",
            "simulation_id": "s1",
            "account": "acc1",
        }
        df_in = pd.DataFrame([row, row])  # exact duplicate
        _write_csv(raw_dir, df_in)

        cleaner = TradingLogCleaner(raw_dir=raw_dir, processed_dir=processed_dir)
        result = cleaner.run()
        assert len(result) == 1  # deduplicated

    def test_group_nearby_trades(self, temp_dirs):
        raw_dir, processed_dir = temp_dirs
        # Two trades within 500ms for the same account/strategy
        df_in = pd.DataFrame(
            {
                "timestamp": [
                    "2024-01-15 10:00:00.000+00:00",
                    "2024-01-15 10:00:00.300+00:00",  # 300 ms later
                ],
                "direction": ["buy", "buy"],
                "price": [100.0, 101.0],
                "quantity": [10, 20],
                "pnl": [5.0, 3.0],
                "strategy_permutation": ["a", "a"],
                "simulation_id": ["s1", "s1"],
                "account": ["acc1", "acc1"],
            }
        )
        _write_csv(raw_dir, df_in)

        cleaner = TradingLogCleaner(
            raw_dir=raw_dir, processed_dir=processed_dir, group_ms=500
        )
        result = cleaner.run()
        # Should be grouped into 1 row (quantity summed, price averaged)
        assert len(result) == 1
        assert result.iloc[0]["quantity"] == 30
        assert result.iloc[0]["pnl"] == 8.0

    def test_missing_value_dropped(self, temp_dirs):
        raw_dir, processed_dir = temp_dirs
        df_in = pd.DataFrame(
            {
                "timestamp": ["2024-01-15 10:00:00+00:00", None],
                "direction": ["buy", "sell"],
                "price": [100.0, None],
                "quantity": [10, 20],
                "pnl": [5.0, -3.0],
                "strategy_permutation": ["a", "a"],
                "simulation_id": ["s1", "s1"],
                "account": ["acc1", "acc1"],
            }
        )
        _write_csv(raw_dir, df_in)

        cleaner = TradingLogCleaner(raw_dir=raw_dir, processed_dir=processed_dir)
        result = cleaner.run()
        assert len(result) == 1  # second row dropped
