"""Tests for the prediction module."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from crowdwisdom_quant.models.predictor import Predictor


class TestPredictor:
    """Verify that Predictor produces correct-shaped outputs."""

    def test_predict_without_model_raises(self) -> None:
        p = Predictor()
        with pytest.raises(RuntimeError, match="No model"):
            p.predict(np.array([[1.0, 2.0]]))

    def test_predict_after_setting_model(self) -> None:
        X = np.random.randn(10, 3).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        model = XGBRegressor(n_estimators=2, max_depth=2, random_state=42)
        model.fit(X, y)

        p = Predictor()
        p.model = model
        predictions = p.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (10,)
        assert not np.any(np.isnan(predictions))

    def test_predict_with_dataframe(self) -> None:
        X = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.randn(10))
        model = XGBRegressor(n_estimators=2, max_depth=2, random_state=42)
        model.fit(X, y)

        p = Predictor()
        p.model = model
        predictions = p.predict(X)
        assert len(predictions) == 10
