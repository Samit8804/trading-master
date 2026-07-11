"""Model registry for versioning, storing, and retrieving trained models.

Provides a simple file-based registry that stores:

* Model artifacts (``.joblib``) with version tags
* Training metadata (hyperparameters, features, metrics)
* A registry index (``registry.json``) for lookup

Usage::

    from crowdwisdom_quant.models.registry import ModelRegistry

    registry = ModelRegistry()
    registry.register(model, metrics={"rmse": 10.5}, params={"max_depth": 6})
    # → stores model at models/registry/20260711_120000_abc/artifacts/model.joblib

    # List all registered models
    for entry in registry.list():
        print(entry["version"], entry["timestamp"], entry["metrics"])

    # Load a specific version
    model = registry.load("20260711_120000_abc")
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

logger = logging.getLogger(__name__)


class ModelRegistry:
    """File-based model versioning registry.

    Parameters
    ----------
    registry_dir : Path, optional
        Root directory for the registry.  Defaults to
        ``{project_root}/models/registry``.
    """

    def __init__(self, registry_dir: Optional[Path] = None) -> None:
        from crowdwisdom_quant.config.settings import Config
        self._root = (
            registry_dir
            or Config.PROJECT_ROOT / "models" / "registry"
        )
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / "registry.json"
        self._index: List[Dict[str, Any]] = self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model: Any,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a trained model in the registry.

        Parameters
        ----------
        model : Any
            Trained model object (must be joblib-serialisable).
        metrics : dict, optional
            Evaluation metrics (e.g. ``{"rmse": 10.5, "r2": 0.3}``).
        params : dict, optional
            Model hyperparameters.
        feature_names : list of str, optional
            Names of features used by the model.
        tags : dict, optional
            Arbitrary tags for filtering (e.g. ``{"fold": "0"}``).

        Returns
        -------
        str
            Version identifier for the registered model.
        """
        version = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
        run_dir = self._root / version
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save model artifact
        model_path = artifacts_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Build metadata
        meta: Dict[str, Any] = {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics or {},
            "params": params or {},
            "feature_names": feature_names or [],
            "tags": tags or {},
            "artifact_path": str(model_path),
        }

        # Save metadata
        meta_path = run_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Update index
        self._index.append(meta)
        self._save_index()

        logger.info(
            "Registered model version=%s  (metrics=%s)",
            version, metrics,
        )
        return version

    def load(self, version: str) -> Any:
        """Load a model by version identifier.

        Parameters
        ----------
        version : str
            Version string (e.g. ``"20260711_120000_abc"``).

        Returns
        -------
        object
            Deserialised model.

        Raises
        ------
        FileNotFoundError
            If the version does not exist.
        """
        model_path = self._root / version / "artifacts" / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model version '{version}' not found at {model_path}"
            )
        logger.debug("Loading model version=%s", version)
        return joblib.load(model_path)

    def load_latest(self) -> Any:
        """Load the most recently registered model."""
        if not self._index:
            raise ValueError("Registry is empty — no models registered.")
        latest = self._index[-1]
        return self.load(latest["version"])

    def list(
        self,
        sort_by: str = "timestamp",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List registered models, most recent first.

        Parameters
        ----------
        sort_by : str
            Field to sort by (``"timestamp"``, ``"version"``).
        limit : int
            Maximum number of entries.

        Returns
        -------
        list of dict
        """
        entries = sorted(
            self._index,
            key=lambda e: e.get(sort_by, ""),
            reverse=True,
        )
        return entries[:limit]

    def get_metadata(self, version: str) -> Dict[str, Any]:
        """Get metadata for a specific version."""
        meta_path = self._root / version / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata for version '{version}' not found."
            )
        with open(meta_path, "r") as f:
            return json.load(f)

    def delete(self, version: str) -> None:
        """Delete a model version and its artifacts."""
        run_dir = self._root / version
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Version '{version}' does not exist."
            )
        shutil.rmtree(run_dir)
        self._index = [e for e in self._index if e["version"] != version]
        self._save_index()
        logger.info("Deleted model version=%s", version)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> List[Dict[str, Any]]:
        if self._index_path.exists():
            with open(self._index_path, "r") as f:
                return json.load(f)
        return []

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)
