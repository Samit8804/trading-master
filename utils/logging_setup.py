"""Structured logging setup for CrowdWisdomQuant.

Provides:
* ``setup_logging()`` — one-call configuration of loguru or stdlib logging
* JSON-structured output for production / file ingestion
* Console output with color for development
* Automatic log rotation

Usage::

    from crowdwisdom_quant.utils.logging_setup import setup_logging
    setup_logging()

    # Then use the standard logger anywhere:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Pipeline started", extra={"step": "scrape"})
"""

from __future__ import annotations

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    fmt: str = "structured",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """Configure logging for the entire application.

    Parameters
    ----------
    level : str
        Log level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
    fmt : str
        ``"structured"`` for JSON-like output, ``"plain"`` for standard
        ``LEVEL:module:message`` format.
    log_file : str, optional
        Path to log file.  If ``None``, logs go to stderr only.
    rotation : str
        Max size before rotating (e.g. ``"10 MB"``).  Only applies when
        *log_file* is set.
    retention : str
        How long to keep rotated logs (e.g. ``"30 days"``).
    """
    _level = getattr(logging, level.upper(), logging.INFO)

    formatters: dict = {
        "structured": {
            "format": (
                "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
                "  |  %(pathname)s:%(lineno)d"
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "plain": {
            "format": (
                "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }

    handlers: dict = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stderr,
            "formatter": fmt if fmt in formatters else "plain",
            "level": _level,
        },
    }

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_path),
            "maxBytes": _parse_size(rotation),
            "backupCount": _parse_retention_days(retention),
            "formatter": "structured",
            "level": _level,
        }

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "root": {
            "level": _level,
            "handlers": list(handlers.keys()),
        },
    })

    # Suppress noisy third-party loggers
    for noisy in ["matplotlib", "apify_client", "urllib3", "xgboost"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug(
        "Logging configured: level=%s, fmt=%s, file=%s",
        level, fmt, log_file,
    )


def _parse_size(size_str: str) -> int:
    """Parse size strings like ``"10 MB"`` to bytes."""
    size_str = size_str.strip().upper()
    if size_str.endswith("MB"):
        return int(float(size_str[:-2].strip()) * 1024 * 1024)
    if size_str.endswith("KB"):
        return int(float(size_str[:-2].strip()) * 1024)
    if size_str.endswith("GB"):
        return int(float(size_str[:-2].strip()) * 1024 * 1024 * 1024)
    return int(size_str)


def _parse_retention_days(retention: str) -> int:
    """Parse retention strings like ``"30 days"`` to backup count."""
    retention = retention.strip().lower()
    if retention.endswith("days"):
        return int(retention.split()[0])
    return int(retention)
