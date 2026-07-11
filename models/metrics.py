"""Performance metrics for evaluating trading signal predictions.

All metrics are computed from arrays of actual and predicted PnL values.

Metric definitions
------------------
* **RMSE** (Root Mean Squared Error) — penalises large errors quadratically.
  Sensitive to outliers. Lower is better.

* **MAE** (Mean Absolute Error) — average absolute prediction error.
  More robust to outliers than RMSE. Lower is better.

* **R²** (Coefficient of Determination) — proportion of variance explained
  by the model. 1.0 is perfect; 0.0 means the model performs no better
  than always predicting the mean.

* **Sharpe Ratio** — risk-adjusted return.  Mean(PnL) / std(PnL) annualised.
  Values > 1.0 are considered good, > 2.0 excellent, > 3.0 outstanding.
  *Assumes trades are sampled at regular intervals; for irregular trades,
   interpret as a unitless risk-adjusted score.*

* **Sortino Ratio** — like Sharpe but penalises only downside volatility
  (negative deviations). Better for strategies where upside variance is
  desirable. Higher is better.

* **Maximum Drawdown** — largest peak-to-trough decline in cumulative PnL.
  Measures the worst realised loss. Lower (less negative) is better.

* **Win Rate** — fraction of trades with PnL > 0.  > 0.5 indicates a
  profitable strategy *on a per-trade basis*.

* **Profit Factor** — gross profits / |gross losses|.  > 1.0 means the
  strategy is profitable overall.  > 2.0 is strong.
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_metrics(
    y_actual: np.ndarray | pd.Series,
    y_predicted: np.ndarray | pd.Series,
    timestamps: np.ndarray | pd.Series | None = None,
) -> Dict[str, float]:
    """Compute a full suite of regression and trading metrics.

    Parameters
    ----------
    y_actual : array-like
        Ground-truth PnL values.
    y_predicted : array-like
        Model-predicted PnL values.
    timestamps : array-like, optional
        Per-trade timestamps.  Required for correct Sharpe / Sortino
        annualisation (trades are resampled to daily frequency).

    Returns
    -------
    dict
        Metric name -> scalar value.
    """
    y_true = np.asarray(y_actual, dtype=np.float64)
    y_pred = np.asarray(y_predicted, dtype=np.float64)
    ts = np.asarray(timestamps) if timestamps is not None else None

    if len(y_true) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
        }

    # --- Regression metrics ---
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # --- Trading metrics (applied to actual PnL) ---
    sharpe = _sharpe_ratio(y_true, timestamps=ts)
    sortino = _sortino_ratio(y_true, timestamps=ts)
    drawdown = _max_drawdown(y_true)
    win_rate = float(np.mean(y_true > 0))
    profit_factor = _profit_factor(y_true)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


def _sharpe_ratio(pnl: np.ndarray, annual_factor: float = 252.0,
                  timestamps: np.ndarray | None = None) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    pnl : np.ndarray
        Per-trade PnL.
    annual_factor : float
        Scaling factor to annualise (252 for daily data).
    timestamps : np.ndarray, optional
        Per-trade timestamps.  When provided, PnL is resampled to daily
        frequency (summed per calendar day) before computing the ratio.
        This gives a statistically correct annualised Sharpe for
        irregularly-spaced trades.

    Notes
    -----
    Without timestamps (backward compat), uses the raw per-trade PnL,
    which overstates the ratio when there are many trades per day.
    """
    if timestamps is not None:
        daily = _resample_to_daily(pnl, timestamps)
        return _sharpe_ratio(daily, annual_factor=annual_factor, timestamps=None)

    if pnl.std() == 0:
        return 0.0
    return float(np.sqrt(annual_factor) * np.mean(pnl) / np.std(pnl))


def _sortino_ratio(pnl: np.ndarray, annual_factor: float = 252.0,
                   timestamps: np.ndarray | None = None) -> float:
    """Annualised Sortino ratio — penalises only downside deviation."""
    if timestamps is not None:
        daily = _resample_to_daily(pnl, timestamps)
        return _sortino_ratio(daily, annual_factor=annual_factor, timestamps=None)

    downside = pnl[pnl < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(np.sqrt(annual_factor) * np.mean(pnl) / downside.std())


def _resample_to_daily(pnl: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Sum PnL per calendar day for correct annualisation."""
    dates = pd.to_datetime(timestamps).normalize()
    daily = pd.Series(pnl, index=dates).groupby(level=0).sum()
    return daily.values


def _max_drawdown(pnl: np.ndarray) -> float:
    """Maximum drawdown as a fraction (0..-1) of peak cumulative PnL.

    Returns a negative decimal (e.g. -0.25 for a 25% peak-to-trough loss).
    """
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    peak = running_max.copy()
    peak[peak == 0] = 1.0  # avoid division by zero
    drawdowns = (cumulative - running_max) / peak
    return float(drawdowns.min())


def _profit_factor(pnl: np.ndarray) -> float:
    """Ratio of gross profits to |gross losses|.

    Returns ``999.0`` as a sentinel when gross_loss = 0 (instead of inf)
    so the value is serialisable (Parquet / JSON cannot store inf).
    """
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    if gross_loss == 0:
        return 999.0 if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def metrics_dataframe(metrics_list: list[Dict[str, float]]) -> pd.DataFrame:
    """Convert a list of per-fold metric dicts to a summary DataFrame."""
    df = pd.DataFrame(metrics_list)
    df["fold"] = df["fold"].astype(str)
    # Compute mean only on numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    mean_row = {k: (df[k].mean() if k in numeric_cols else "") for k in df.columns}
    mean_row["fold"] = "MEAN"
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df
