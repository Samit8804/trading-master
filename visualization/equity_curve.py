"""Equity curve comparison: AI-selected vs. baseline strategy.

Methodology
-----------
* **AI Selected Strategy**: for each trade in the test set, we use the
  model's predicted PnL as the strategy's return.  If the model predicts
  a positive PnL we take the trade, otherwise we skip it (or take the
  opposite side depending on configuration).
* **Baseline Strategy**: buy-and-hold or a simple moving-average crossover
  as a naive benchmark.
* Both equity curves are normalised to start at 1.0 (cumulative product
  of (1 + return) or cumulative sum of PnL).

For simplicity, we use **cumulative PnL** (not returns) since PnL is
already in monetary units.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from crowdwisdom_quant.config import Config

logger = logging.getLogger(__name__)


def generate_equity_curve(
    prediction_table: pd.DataFrame,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Generate and save an equity curve comparing AI vs. Baseline.

    Parameters
    ----------
    prediction_table : pd.DataFrame
        Must contain ``Timestamp``, ``Actual_PnL``, ``Predicted_PnL``,
        sorted chronologically.
    output_path : Path or str, optional
        Where to save the PNG. Defaults to ``visualization/equity_curve.png``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if prediction_table.empty:
        logger.warning("Empty prediction table — cannot generate equity curve.")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    out = Path(output_path or Config.VISUALIZATION_DIR / "equity_curve.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    df = prediction_table.sort_values("Timestamp").reset_index(drop=True)

    # --- Baseline: simply take every trade (cumulative actual PnL) ---
    baseline_equity = df["Actual_PnL"].cumsum()

    # --- AI-Selected strategy: take trade only if model predicts positive PnL ---
    ai_pnl = df["Actual_PnL"].where(df["Predicted_PnL"] > 0, 0)
    ai_equity = ai_pnl.cumsum()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Timestamp"], baseline_equity, label="Baseline (All Trades)", alpha=0.7)
    ax.plot(df["Timestamp"], ai_equity, label="AI-Selected Strategy", alpha=0.9)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_title("Equity Curve: AI-Selected Strategy vs. Baseline")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Cumulative PnL")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Equity curve saved to %s", out)
    return fig
