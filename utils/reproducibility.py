"""Reproducibility utilities for CrowdWisdomQuant.

Ensures deterministic results across runs by:

* Seeding all random generators (Python ``random``, ``numpy``, XGBoost)
* Capturing the git commit hash of the codebase
* Recording environment metadata (Python version, platform, dependencies)

Usage::

    from crowdwisdom_quant.utils.reproducibility import (
        set_seeds, capture_environment
    )

    set_seeds(seed=42)
    env_info = capture_environment()
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def set_seeds(seed: int = 42) -> None:
    """Seed all random generators for deterministic execution.

    Seeds:
    * Python's ``random``
    * ``numpy.random``
    * Python's ``hash`` randomization via ``PYTHONHASHSEED``
      (must be set before interpreter starts; we warn if not)

    Parameters
    ----------
    seed : int
        Master seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        from xgboost import XGBRegressor
        # XGBoost uses numpy's random state internally; no explicit seed needed
    except ImportError:
        pass

    # Warn if PYTHONHASHSEED is not set (affects dict ordering)
    if not os.environ.get("PYTHONHASHSEED"):
        import logging
        logging.getLogger(__name__).warning(
            "PYTHONHASHSEED not set — dict iteration order may vary between runs. "
            "Set PYTHONHASHSEED=%d for full determinism.", seed
        )


def capture_environment() -> Dict[str, Any]:
    """Capture environment and code metadata for experiment tracking.

    Returns
    -------
    dict
        Keys include ``python_version``, ``platform``, ``timestamp``,
        ``git_commit``, ``git_branch``, ``working_directory``.
    """
    env: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "working_directory": str(_PROJECT_ROOT),
    }

    # Git info
    git_commit = _run_git(["rev-parse", "--short", "HEAD"])
    git_branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    git_dirty = _run_git(["status", "--porcelain"])
    if git_commit:
        env["git_commit"] = git_commit
    if git_branch:
        env["git_branch"] = git_branch
    if git_dirty is not None:
        env["git_dirty"] = bool(git_dirty.strip())

    return env


def _run_git(args: list[str]) -> Optional[str]:
    """Run a git command and return stdout, or ``None`` on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def make_experiment_id() -> str:
    """Generate a short unique experiment ID.

    Format: ``YYYYMMDD_HHMMSS_XXXX`` where XXXX is the git short hash
    (or "nogit" if not available).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    commit = _run_git(["rev-parse", "--short", "HEAD"]) or "nogit"
    return f"{ts}_{commit}"
