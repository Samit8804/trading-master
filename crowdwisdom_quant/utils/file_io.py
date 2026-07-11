"""File I/O helpers for the CrowdWisdomQuant pipeline.

Centralises reading and writing of common data formats (Parquet, CSV, JSON)
to ensure consistent behaviour and error handling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Read a Parquet file with error handling."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path)
    logger.debug("Read Parquet: %s (%d rows, %d cols)", path, len(df), len(df.columns))
    return df


def write_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs: Any) -> Path:
    """Write a DataFrame to Parquet, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)
    logger.debug("Wrote Parquet: %s (%d rows, %d cols)", path, len(df), len(df.columns))
    return path


def read_json(path: Union[str, Path]) -> Any:
    """Read a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def write_json(data: Any, path: Union[str, Path], indent: int = 2) -> Path:
    """Write data to a JSON file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)
    return path


def read_csv(
    path: Union[str, Path],
    parse_dates: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file with optional date parsing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path, parse_dates=parse_dates, **kwargs)
    logger.debug("Read CSV: %s (%d rows, %d cols)", path, len(df), len(df.columns))
    return df
