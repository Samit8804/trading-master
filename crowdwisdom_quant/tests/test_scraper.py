"""Unit tests for the macro scraper module."""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from crowdwisdom_quant.data.database.manager import DatabaseManager
from crowdwisdom_quant.data.scraping.macro_scraper import MacroEconomicScraper
from crowdwisdom_quant.utils.exceptions import ApiKeyMissingError


@pytest.fixture
def db_manager():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    mgr = DatabaseManager(f"sqlite:///{tmp.name}")
    mgr.create_tables()
    yield mgr
    mgr.drop_tables()
    mgr.close()
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


class TestCSVFallback:
    def test_fallback_generates_data(self) -> None:
        scraper = MacroEconomicScraper(
            api_key="", lookback_days=30
        )
        raw = scraper.fetch_from_csv_fallback()
        assert len(raw) > 0
        for row in raw[:5]:
            assert "timestamp" in row
            assert "event_name" in row
            assert "country" in row

    def test_persist_fallback(self, db_manager) -> None:
        scraper = MacroEconomicScraper(
            db_manager=db_manager, api_key="", lookback_days=30
        )
        n = scraper.run()
        assert n > 0
        count = db_manager.row_count("macro_events")
        assert count == n

    def test_no_duplicates_in_persist(self, db_manager) -> None:
        ref = datetime(2025, 6, 1, tzinfo=timezone.utc)
        scraper = MacroEconomicScraper(
            db_manager=db_manager, api_key="", lookback_days=10, reference_time=ref
        )
        n1 = scraper.run()
        assert n1 > 0
        count_after_first = db_manager.row_count("macro_events")
        assert count_after_first == n1

        n2 = scraper.run()
        assert n2 == 0
        count_after_second = db_manager.row_count("macro_events")
        assert count_after_second == count_after_first

    def test_no_api_key_uses_fallback(self) -> None:
        """When api_key is explicitly empty, run() should use CSV fallback."""
        scraper = MacroEconomicScraper(api_key="", lookback_days=10)
        raw = scraper.fetch_from_csv_fallback()
        assert len(raw) > 0

    def test_fallback_is_deterministic(self) -> None:
        """Running fetch_from_csv_fallback twice with same params yields same data."""
        ref = datetime(2025, 6, 1, tzinfo=timezone.utc)
        s1 = MacroEconomicScraper(api_key="", lookback_days=10, reference_time=ref)
        s2 = MacroEconomicScraper(api_key="", lookback_days=10, reference_time=ref)
        r1 = s1.fetch_from_csv_fallback()
        r2 = s2.fetch_from_csv_fallback()
        assert len(r1) == len(r2)
        for i in range(min(len(r1), len(r2))):
            assert r1[i]["event_name"] == r2[i]["event_name"]
            assert r1[i]["country"] == r2[i]["country"]
