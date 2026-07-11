"""Apify-based scraper for macro-economic calendar events.

Downloads the last 180 days of major economic announcements including CPI,
FOMC announcements, employment reports, interest rate decisions, and GDP
releases.  The scraper is split into a ``fetch_from_api()`` method (live
Apify actor) and a ``fetch_from_csv_fallback()`` method for development / CI
when no API key is available.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from apify_client.errors import ApifyApiError

from crowdwisdom_quant.config.settings import Config
from crowdwisdom_quant.data.database.manager import DatabaseManager
from crowdwisdom_quant.data.database.schema import MacroEvent
from crowdwisdom_quant.utils.exceptions import (
    ApiKeyMissingError,
    ApiRateLimitError,
    ApiResponseError,
    ScraperError,
)
from crowdwisdom_quant.utils.retry import retry

logger = logging.getLogger(__name__)

# Filter keywords that identify target event types in the actor output.
# Each list contains substrings that would appear in the event name or
# category field of the scraped data.
_EVENT_FILTERS: Dict[str, List[str]] = {
    "CPI": ["cpi", "consumer price index", "consumer prices"],
    "FOMC": ["fomc", "fed minutes", "monetary policy"],
    "Employment": [
        "non-farm payrolls",
        "unemployment rate",
        "employment",
        "jobless claims",
    ],
    "Interest Rate": ["interest rate decision", "fed rate", "rate decision"],
    "GDP": ["gdp", "gross domestic product"],
}


class MacroEconomicScraper:
    """Scrapes macro-economic calendar events and persists them to SQLite."""

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        api_key: str | None = None,
        lookback_days: int | None = None,
        reference_time: datetime | None = None,
    ) -> None:
        self.db = db_manager or DatabaseManager()
        self.api_key = api_key if api_key is not None else Config.APIFY_API_KEY
        self.lookback_days = lookback_days or Config.SCRAPE_LOOKBACK_DAYS
        self._reference_time = reference_time  # for test reproducibility
        self._client: Any = None  # lazy ApifyClient

    # ------------------------------------------------------------------
    # Apify client
    # ------------------------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None and self.api_key:
            from apify_client import ApifyClient

            self._client = ApifyClient(self.api_key)
        return self._client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def run(self) -> int:
        """Scrape macro events and store them in the database.

        Returns
        -------
        int
            Number of events inserted (not skipped as duplicates).

        Raises
        ------
        ApiKeyMissingError
            No API key and no CSV fallback produced data.
        ScraperError
            All API retry attempts exhausted.
        """
        self.db.create_tables()

        try:
            if self.client is not None:
                logger.info("Fetching data via Apify API ...")
                raw = self.fetch_from_api()
            else:
                logger.info(
                    "No Apify API key found – using CSV fallback. "
                    "Set APIFY_API_KEY environment variable for live data."
                )
                raw = self.fetch_from_csv_fallback()
        except (requests.RequestException, ApifyApiError) as exc:
            raise ScraperError(
                f"Macro scraper failed after all retries: {exc}"
            ) from exc

        if not raw:
            raise ApiKeyMissingError(
                "No macro data could be fetched. "
                "Set APIFY_API_KEY or verify the CSV fallback."
            )

        events = self._normalise(raw)
        inserted = self._persist(events)
        logger.info("Inserted %d / %d macro events.", len(inserted), len(events))
        return len(inserted)

    # ------------------------------------------------------------------
    # Fetch implementations
    # ------------------------------------------------------------------
    @retry(
        max_attempts=3,
        base_delay=1.0,
        exceptions=(requests.RequestException, ApifyApiError, ApiResponseError),
    )
    def fetch_from_api(self) -> List[Dict[str, Any]]:
        """Call the Apify actor and return raw event dicts.

        Retries up to 3 times with exponential back-off on failure.
        """
        if self.client is None:
            raise ApiKeyMissingError(
                "ApifyClient not available – set APIFY_API_KEY environment variable."
            )

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        actor_id = Config.APIFY_MACRO_DATASET_ID

        logger.debug("Calling Apify actor %s ...", actor_id)
        run_result = self.client.actor(actor_id).call(
            run_input={
                "startDate": cutoff.strftime("%Y-%m-%d"),
                "endDate": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }
        )
        dataset_id = run_result.get("defaultDatasetId")
        if not dataset_id:
            raise ApiResponseError(
                f"Actor {actor_id} returned no defaultDatasetId"
            )

        items = self.client.dataset(dataset_id).list_items().items
        logger.debug("Apify returned %d raw items.", len(items))
        return list(items)

    def fetch_from_csv_fallback(self) -> List[Dict[str, Any]]:
        """Generate synthetic but realistic macro events for development.

        This fallback produces deterministic data so that the pipeline can
        be exercised without an Apify API key.
        """
        logger.warning("Using CSV fallback with synthetic macro data.")

        rows: List[Dict[str, Any]] = []
        now = self._reference_time or datetime.now(timezone.utc)
        base = now - timedelta(days=self.lookback_days)

        event_templates = [
            ("CPI", "United States", "high"),
            ("FOMC Statement", "United States", "high"),
            ("Non-Farm Payrolls", "United States", "high"),
            ("Interest Rate Decision", "United States", "high"),
            ("GDP", "United States", "high"),
            ("CPI", "Eurozone", "medium"),
            ("Employment Change", "Eurozone", "medium"),
        ]

        for day_offset in range(self.lookback_days):
            for name, country, importance in event_templates:
                seed = int.from_bytes(hashlib.md5(name.encode()).digest()[:4], "big")
                ts = base + timedelta(days=day_offset, hours=seed % 12 + 8)
                # Publish at most one event per category per week for realism
                if day_offset % 7 != seed % 7:
                    continue
                rows.append(
                    {
                        "timestamp": ts.isoformat(),
                        "event_name": name,
                        "country": country,
                        "forecast": round(0.1 + seed % 100 / 10, 1),
                        "actual": round(0.1 + (seed + day_offset) % 100 / 10, 1),
                        "previous": round(0.1 + (seed - 1) % 100 / 10, 1),
                        "importance": importance,
                        "timezone": "UTC",
                    }
                )

        return rows

    # ------------------------------------------------------------------
    # Normalisation (private helpers)
    # ------------------------------------------------------------------
    def _normalise(
        self, raw: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse timestamps to UTC and keep only targeted event types.

        The Apify actor may return many hundreds of calendar rows; we
        filter down to the event families defined in ``_EVENT_FILTERS``.
        """
        normalised: List[Dict[str, Any]] = []

        for row in raw:
            ts_raw = row.get("timestamp") or row.get("date") or row.get("time")
            if ts_raw is None:
                continue

            ts = pd.to_datetime(ts_raw, utc=True)
            if ts is pd.NaT:
                continue

            name = str(row.get("event_name", row.get("title", "")))
            if not self._is_target_event(name):
                continue

            normalised.append(
                {
                    "timestamp": ts.to_pydatetime(),
                    "event_name": name.strip(),
                    "country": str(row.get("country", "")),
                    "forecast": self._safe_float(row.get("forecast")),
                    "actual": self._safe_float(row.get("actual")),
                    "previous": self._safe_float(row.get("previous")),
                    "importance": str(row.get("importance", "low")),
                    "timezone": str(row.get("timezone", "UTC")),
                }
            )

        return normalised

    @staticmethod
    def _is_target_event(event_name: str) -> bool:
        name_lower = event_name.lower()
        for keywords in _EVENT_FILTERS.values():
            for kw in keywords:
                if kw in name_lower:
                    return True
        return False

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _persist(self, events: List[Dict[str, Any]]) -> List[MacroEvent]:
        """Insert normalised events into the database."""
        orm_objects = [
            MacroEvent(
                timestamp=ev["timestamp"],
                event_name=ev["event_name"],
                country=ev["country"],
                forecast=ev["forecast"],
                actual=ev["actual"],
                previous=ev["previous"],
                importance=ev["importance"],
                timezone=ev["timezone"],
            )
            for ev in events
        ]

        with self.db.session() as session:
            return self.db.bulk_insert_macro_events(session, orm_objects)
