"""End-to-end simulation runner: book + engine + flow + state capture."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd

from lob.matching_engine import MatchingEngine, Trade
from lob.order_book import OrderBook
from lob.order_flow import FlowEvent, OrderFlowParams, OrderFlowSimulator
from lob.strategy import Strategy


StrategyFactory = Callable[[MatchingEngine, OrderBook], Strategy]


@dataclass
class BookSnapshot:
    """Point-in-time book state for time-series analysis."""
    timestamp: int
    mid: Optional[float]
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    bid_volume_l1: int
    ask_volume_l1: int
    bid_levels: int
    ask_levels: int
    resting_orders: int


@dataclass
class StrategySnapshot:
    """Point-in-time strategy state for P&L analysis."""
    timestamp: int
    inventory: int
    cash: float
    pnl: Optional[float]  # MTM at current mid; None when no mid available
    n_fills: int
    n_active_orders: int


@dataclass
class SimulationStats:
    """Aggregate counters for a completed (or running) simulation."""
    events_processed: int
    new_orders: int
    cancels_attempted: int
    cancels_succeeded: int
    trades: int
    total_volume: int
    elapsed_ns: int


class Simulation:
    """Owns book + engine + simulator + optional strategy.

    If a ``strategy_factory`` is provided, the strategy is built with the
    Simulation's engine and book, called after every market event, and
    snapshotted alongside the book.
    """

    def __init__(
        self,
        params: Optional[OrderFlowParams] = None,
        strategy_factory: Optional[StrategyFactory] = None,
        snapshot_interval_ns: int = 100_000_000,
        seed: Optional[int] = None,
    ) -> None:
        self.params = params or OrderFlowParams()
        self.snapshot_interval_ns = snapshot_interval_ns
        self.book = OrderBook()
        self.engine = MatchingEngine(self.book)

        self.strategy: Optional[Strategy] = (
            strategy_factory(self.engine, self.book) if strategy_factory else None
        )

        # Strategy orders are protected from being randomly cancelled by
        # the noise simulator — only the strategy itself manages them.
        is_protected = (
            (lambda oid: oid in self.strategy.active_orders)
            if self.strategy is not None else None
        )

        self.simulator = OrderFlowSimulator(
            params=self.params,
            book=self.book,
            rng=random.Random(seed),
            is_protected=is_protected,
        )

        self.trades: List[Trade] = []
        self.snapshots: List[BookSnapshot] = []
        self.strategy_snapshots: List[StrategySnapshot] = []

        self._n_events_processed = 0
        self._n_new_orders = 0
        self._n_cancels_attempted = 0
        self._n_cancels_succeeded = 0
        self._last_snapshot_time = 0
        self._last_known_mid: Optional[float] = None

        # initial snapshot
        self._snapshot(self.simulator.current_time)

    # ---- public API --------------------------------------------------------

    @property
    def events_processed(self) -> int:
        return self._n_events_processed

    def run(self, n_events: int) -> SimulationStats:
        if n_events < 0:
            raise ValueError("n_events must be non-negative")
        for event in self.simulator.generate(n_events):
            self._handle_event(event)
            self._n_events_processed += 1
            self._maybe_snapshot()
        self._snapshot(self.simulator.current_time)
        return self.stats()

    def stats(self) -> SimulationStats:
        return SimulationStats(
            events_processed=self._n_events_processed,
            new_orders=self._n_new_orders,
            cancels_attempted=self._n_cancels_attempted,
            cancels_succeeded=self._n_cancels_succeeded,
            trades=len(self.trades),
            total_volume=sum(t.quantity for t in self.trades),
            elapsed_ns=self.simulator.current_time,
        )

    def trades_df(self) -> pd.DataFrame:
        cols = [
            "trade_id", "timestamp", "price", "quantity",
            "aggressor_side", "aggressor_order_id", "maker_order_id",
        ]
        if not self.trades:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([
            {
                "trade_id": t.trade_id,
                "timestamp": t.timestamp,
                "price": t.price,
                "quantity": t.quantity,
                "aggressor_side": t.aggressor_side.value,
                "aggressor_order_id": t.aggressor_order_id,
                "maker_order_id": t.maker_order_id,
            }
            for t in self.trades
        ])

    def snapshots_df(self) -> pd.DataFrame:
        if not self.snapshots:
            return pd.DataFrame()
        return pd.DataFrame([s.__dict__ for s in self.snapshots])

    def strategy_snapshots_df(self) -> pd.DataFrame:
        if not self.strategy_snapshots:
            return pd.DataFrame()
        return pd.DataFrame([s.__dict__ for s in self.strategy_snapshots])

    # ---- internals ---------------------------------------------------------

    def _handle_event(self, event: FlowEvent) -> None:
        trades_from_event: List[Trade] = []
        if event.is_cancel:
            self._n_cancels_attempted += 1
            try:
                self.engine.cancel(event.cancel_order_id)
                self._n_cancels_succeeded += 1
            except KeyError:
                pass
        else:
            self._n_new_orders += 1
            trades_from_event = self.engine.submit(event.order)
            self.trades.extend(trades_from_event)

        # Notify the strategy after the engine has processed the event
        if self.strategy is not None:
            self.strategy.on_event(
                trades=trades_from_event,
                current_time=self.simulator.current_time,
            )

    def _maybe_snapshot(self) -> None:
        now = self.simulator.current_time
        if now - self._last_snapshot_time >= self.snapshot_interval_ns:
            self._snapshot(now)

    def _snapshot(self, ts: int) -> None:
        bb_level = self.book.best_bid()
        ba_level = self.book.best_ask()
        mid = self.book.mid_price
        if mid is not None:
            self._last_known_mid = mid

        self.snapshots.append(BookSnapshot(
            timestamp=ts,
            mid=mid,
            best_bid=self.book.best_bid_price,
            best_ask=self.book.best_ask_price,
            spread=self.book.spread,
            bid_volume_l1=bb_level.total_volume if bb_level else 0,
            ask_volume_l1=ba_level.total_volume if ba_level else 0,
            bid_levels=self.book.bid_count,
            ask_levels=self.book.ask_count,
            resting_orders=self.book.total_orders,
        ))
        self._last_snapshot_time = ts

        if self.strategy is not None:
            mark = self._last_known_mid or self.params.reference_price
            pos = self.strategy.position
            self.strategy_snapshots.append(StrategySnapshot(
                timestamp=ts,
                inventory=pos.inventory,
                cash=pos.cash,
                pnl=pos.mtm(mark),
                n_fills=len(self.strategy.fills),
                n_active_orders=len(self.strategy.active_orders),
            ))