"""SQLAlchemy ORM models for the CrowdWisdomTrading database.

This module defines the two core ORM tables used throughout the pipeline:

  * ``macro_events``  – economic indicator announcements scraped via Apify.
  * ``trading_logs``  – simulated trade entries for strategy evaluation.

All timestamps are normalised to UTC and stored as ISO-8601 text to remain
database-agnostic (SQLite has no native datetime type).  When migrating to
PostgreSQL the column types can be switched to TIMESTAMP WITH TIME ZONE.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, Float, String, DateTime, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class MacroEvent(Base):
    """Economic calendar event (CPI, FOMC, Employment, etc.)."""

    __tablename__ = "macro_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    event_name = Column(String(255), nullable=False)
    country = Column(String(64), nullable=True)
    forecast = Column(Float, nullable=True)
    actual = Column(Float, nullable=True)
    previous = Column(Float, nullable=True)
    importance = Column(String(16), nullable=True)  # e.g. "high", "medium"
    timezone = Column(String(64), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("timestamp", "event_name", name="uq_macro_event"),
    )

    def __repr__(self) -> str:
        return (
            f"<MacroEvent(id={self.id}, event='{self.event_name}', "
            f"ts={self.timestamp})>"
        )


class TradingLog(Base):
    """Individual trade record produced by a simulation run."""

    __tablename__ = "trading_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    account = Column(String(64), nullable=False)
    direction = Column(String(8), nullable=False)  # "buy" or "sell"
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    pnl = Column(Float, nullable=False)
    strategy_permutation = Column(String(128), nullable=False)
    simulation_id = Column(String(64), nullable=False)
    timezone = Column(String(64), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<TradingLog(id={self.id}, sim={self.simulation_id}, "
            f"strat={self.strategy_permutation}, pnl={self.pnl:.2f})>"
        )
