"""Heatmap generation: day-of-week × hour grid showing the best strategy.

For each (Day, Hour) cell, we select the strategy permutation that has
the **highest mean predicted PnL** across all test-fold predictions.
This reveals time-based patterns in strategy performance.

The heatmap is saved as a PNG image.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

from crowdwisdom_quant.config import Config

logger = logging.getLogger(__name__)

# Ordered day-of-week labels
DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def generate_heatmap(
    prediction_table: pd.DataFrame,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Generate and save a (Day × Hour) heatmap of best predicted strategy.

    Parameters
    ----------
    prediction_table : pd.DataFrame
        Must contain columns ``Hour``, ``Day``, ``Strategy``, ``Predicted_PnL``.
    output_path : Path or str, optional
        Where to save the PNG. Defaults to ``visualization/heatmap.png``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if prediction_table.empty:
        logger.warning("Empty prediction table — cannot generate heatmap.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    out = Path(output_path or Config.VISUALIZATION_DIR / "heatmap.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Pivot: Day (rows) × Hour (cols) → best strategy by mean Predicted PnL
    idx = prediction_table.groupby(["Day", "Hour"])["Predicted_PnL"].idxmax()
    grouped = prediction_table.loc[idx, ["Day", "Hour", "Strategy"]].reset_index(drop=True)
    grouped = grouped.rename(columns={"Strategy": "Best_Strategy"})

    # Pivot to matrix
    pivot = grouped.pivot_table(
        index="Day",
        columns="Hour",
        values="Best_Strategy",
        aggfunc="first",
    )

    # Reindex rows to canonical day order
    pivot = pivot.reindex(
        [d for d in DAY_ORDER if d in pivot.index]
    )

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    # Use a simple categorical colormap — we create a numeric backing
    # and annotate with strategy names
    unique_strategies = pivot.values.flatten()
    unique_strategies = unique_strategies[unique_strategies != "N/A"]
    # Filter out non-string entries (e.g. NaN floats from empty cells)
    strategies = sorted({s for s in unique_strategies if isinstance(s, str)})
    strategy_to_idx = {s: i for i, s in enumerate(strategies)}

    numeric_data = pivot.map(lambda v: strategy_to_idx.get(v, -1))

    cmap = sns.color_palette("Set2", n_colors=max(1, len(strategies)))
    sns.heatmap(
        numeric_data,
        annot=pivot.values,
        fmt="",
        cmap=cmap,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"ticks": list(strategy_to_idx.values())},
    )
    colorbar = ax.collections[0].colorbar
    if colorbar and strategies:
        colorbar.set_ticks(list(strategy_to_idx.values()))
        colorbar.set_ticklabels(strategies)

    ax.set_title("Best Strategy by Day of Week and Hour (Predicted PnL)")
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Day of Week")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Heatmap saved to %s", out)
    return fig
