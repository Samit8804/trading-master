"""Unit tests for the database module."""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from crowdwisdom_quant.database.db import DatabaseManager
from crowdwisdom_quant.database.schema import MacroEvent, TradingLog


@pytest.fixture
def db_manager():
    """Create a DatabaseManager backed by a temporary SQLite file."""
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
        logger = __import__("logging").getLogger(__name__)
        logger.warning("Could not delete temp DB file (Windows lock): %s", tmp.name)


class TestMacroEvent:
    def test_insert_and_query(self, db_manager):
        with db_manager.session() as s:
            ev = MacroEvent(
                timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
                event_name="CPI",
                country="US",
                forecast=0.3,
                actual=0.35,
                previous=0.2,
                importance="high",
            )
            inserted = db_manager.bulk_insert_macro_events(s, [ev])
            assert len(inserted) == 1

            results = db_manager.get_macro_events(s)
            assert len(results) == 1
            assert results[0].event_name == "CPI"

    def test_no_duplicate_insert(self, db_manager):
        ev1 = MacroEvent(
            timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            event_name="CPI",
            country="US",
            forecast=0.3,
            actual=0.35,
            previous=0.2,
            importance="high",
        )
        ev2 = MacroEvent(
            timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            event_name="CPI",
            country="US",
            forecast=0.3,
            actual=0.35,
            previous=0.2,
            importance="high",
        )
        # First insert
        with db_manager.session() as s:
            r1 = db_manager.bulk_insert_macro_events(s, [ev1])
            assert len(r1) == 1
        # Second insert (duplicate) in new session
        with db_manager.session() as s:
            r2 = db_manager.bulk_insert_macro_events(s, [ev2])
            assert len(r2) == 0  # duplicate skipped
        assert db_manager.row_count("macro_events") == 1


class TestTradingLog:
    def test_bulk_insert(self, db_manager):
        logs = [
            TradingLog(
                timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
                account="test1",
                direction="buy",
                quantity=100,
                price=150.0,
                pnl=50.0,
                strategy_permutation="strat_a",
                simulation_id="sim_1",
            ),
            TradingLog(
                timestamp=datetime(2024, 1, 15, 10, 5, tzinfo=timezone.utc),
                account="test1",
                direction="sell",
                quantity=50,
                price=151.0,
                pnl=-20.0,
                strategy_permutation="strat_b",
                simulation_id="sim_1",
            ),
        ]
        with db_manager.session() as s:
            inserted = db_manager.bulk_insert_trading_logs(s, logs)
            assert len(inserted) == 2

        with db_manager.session() as s:
            results = db_manager.get_trading_logs(s)
            assert len(results) == 2

    def test_duplicate_skip(self, db_manager):
        # Use fresh objects to avoid DetachedInstanceError
        def _make_log():
            return TradingLog(
                timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
                account="test1",
                direction="buy",
                quantity=100,
                price=150.0,
                pnl=50.0,
                strategy_permutation="strat_a",
                simulation_id="sim_1",
            )
        with db_manager.session() as s:
            inserted = db_manager.bulk_insert_trading_logs(s, [_make_log()])
            assert len(inserted) == 1  # first time
        with db_manager.session() as s:
            inserted = db_manager.bulk_insert_trading_logs(s, [_make_log()])
            assert len(inserted) == 0  # duplicate skipped
