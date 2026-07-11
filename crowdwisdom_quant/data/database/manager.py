"""Database engine, session management, and convenience CRUD helpers.

This module provides the ``DatabaseManager`` class that owns a SQLAlchemy engine
and session factory.  All write operations use ``flush()`` rather than
``commit()`` so the caller can batch transactions in an outer context manager.
The connection URL is read from ``Config.DATABASE_URL`` to make migration from
SQLite to PostgreSQL a one-line change.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, List, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from crowdwisdom_quant.config.settings import Config
from crowdwisdom_quant.data.database.schema import Base, MacroEvent, TradingLog

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages the database engine, session lifecycle, and common queries."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or Config.DATABASE_URL or "sqlite://"
        self.engine = create_engine(
            self.database_url,
            echo=False,
            connect_args={"check_same_thread": False}  # needed for SQLite
            if "sqlite" in self.database_url
            else {},
        )

        # Enable WAL mode for SQLite (concurrent reads without blocking writes)
        if "sqlite" in self.database_url:

            @event.listens_for(self.engine, "connect")
            def _set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA foreign_keys=ON;")
                cursor.close()

        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------
    def create_tables(self) -> None:
        """Create all tables defined in the ORM metadata."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self) -> None:
        """Drop all tables (useful in tests)."""
        Base.metadata.drop_all(bind=self.engine)

    # ------------------------------------------------------------------
    # Session context
    # ------------------------------------------------------------------
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations.

        Usage::

            with db.session() as s:
                s.add(some_object)
                # auto-commits on success, rolls back on exception
        """
        session: Session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Macro-event helpers
    # ------------------------------------------------------------------
    def bulk_insert_macro_events(
        self, session: Session, events: List[MacroEvent]
    ) -> List[MacroEvent]:
        """Insert many macro events, skipping duplicates.  Returns only newly inserted rows.

        Duplicate detection relies on the ``uq_macro_event`` unique constraint
        on ``(timestamp, event_name)``.  Constraint violations are silently skipped
        using savepoints so the outer transaction is preserved.
        """
        inserted: List[MacroEvent] = []
        for ev in events:
            try:
                with session.begin_nested():
                    session.add(ev)
                inserted.append(ev)
            except Exception:
                pass
        session.flush()
        return inserted

    def get_macro_events(
        self,
        session: Session,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> List[MacroEvent]:
        """Query macro events, optionally filtered by time range."""
        q = session.query(MacroEvent)
        if start is not None:
            q = q.filter(MacroEvent.timestamp >= start)
        if end is not None:
            q = q.filter(MacroEvent.timestamp <= end)
        return q.order_by(MacroEvent.timestamp).all()

    # ------------------------------------------------------------------
    # Trading-log helpers
    # ------------------------------------------------------------------
    def bulk_insert_trading_logs(
        self, session: Session, logs: List[TradingLog]
    ) -> List[TradingLog]:
        """Insert many trading logs, skipping exact duplicates.  Returns only newly inserted."""
        inserted: List[TradingLog] = []
        for log in logs:
            existing: Optional[TradingLog] = (
                session.query(TradingLog)
                .filter(
                    TradingLog.timestamp == log.timestamp,
                    TradingLog.account == log.account,
                    TradingLog.simulation_id == log.simulation_id,
                    TradingLog.strategy_permutation == log.strategy_permutation,
                )
                .first()
            )
            if existing is not None:
                continue
            session.add(log)
            inserted.append(log)
        session.flush()
        return inserted

    def get_trading_logs(
        self,
        session: Session,
        start: datetime | None = None,
        end: datetime | None = None,
        accounts: List[str] | None = None,
    ) -> List[TradingLog]:
        """Query trading logs with optional filters."""
        q = session.query(TradingLog)
        if start is not None:
            q = q.filter(TradingLog.timestamp >= start)
        if end is not None:
            q = q.filter(TradingLog.timestamp <= end)
        if accounts:
            q = q.filter(TradingLog.account.in_(accounts))
        return q.order_by(TradingLog.timestamp).all()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def row_count(self, table_name: str, session: Session | None = None) -> int:
        """Return the current row count for a given table.

        If a *session* is provided, the count is run inside that transaction
        (visible uncommitted writes).  Otherwise a new session is opened.
        """
        if session is not None:
            if table_name == "macro_events":
                return session.query(MacroEvent).count()
            elif table_name == "trading_logs":
                return session.query(TradingLog).count()
            return 0
        with self.session() as s:
            if table_name == "macro_events":
                return s.query(MacroEvent).count()
            elif table_name == "trading_logs":
                return s.query(TradingLog).count()
            return 0

    def close(self) -> None:
        """Dispose of the engine connection pool (use in tests for cleanup)."""
        self.engine.dispose()
