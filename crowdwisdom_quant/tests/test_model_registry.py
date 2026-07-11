"""Tests for the model registry."""

from pathlib import Path

import pytest
from xgboost import XGBRegressor

from crowdwisdom_quant.models.registry import ModelRegistry


class TestModelRegistry:
    """Verify model version registration and retrieval."""

    def test_register_and_list(self) -> None:
        registry = ModelRegistry()
        model = XGBRegressor(n_estimators=2, max_depth=2, random_state=42)
        version = registry.register(
            model,
            metrics={"rmse": 10.5, "r2": -0.05},
            params={"max_depth": 4},
            feature_names=["feat1", "feat2"],
        )
        entries = registry.list()
        assert len(entries) >= 1
        matching = [e for e in entries if e["version"] == version]
        assert len(matching) == 1
        assert matching[0]["metrics"]["rmse"] == 10.5

    def test_load_nonexistent_version(self) -> None:
        registry = ModelRegistry()
        with pytest.raises((KeyError, FileNotFoundError)):
            registry.load("nonexistent_version")

    def test_register_multiple_versions(self) -> None:
        registry = ModelRegistry()
        m = XGBRegressor(n_estimators=2, max_depth=2, random_state=42)
        v1 = registry.register(m, metrics={"rmse": 10.0}, params={}, feature_names=[])
        v2 = registry.register(m, metrics={"rmse": 9.0}, params={}, feature_names=[])
        assert v1 != v2
        entries = registry.list()
        assert len(entries) >= 2

    def test_load_returns_model(self) -> None:
        registry = ModelRegistry()
        model = XGBRegressor(n_estimators=2, max_depth=2, random_state=42)
        version = registry.register(model, metrics={}, params={}, feature_names=[])
        loaded = registry.load(version)
        assert loaded is not None

    def test_list_returns_list(self) -> None:
        registry = ModelRegistry()
        result = registry.list()
        assert isinstance(result, list)
