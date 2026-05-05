"""End-to-end simulation runner: book + engine + flow + state capture."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from lob.matching_engine import MatchingEngine, Trade
from lob.order_book import OrderBook
from lob.order_flow import FlowEvent, OrderFlowParams, OrderFlowSimulator


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
    """Owns book + engine + simulator and captures trades + snapshots.

    Time is measured in simulated nanoseconds, driven by the simulator's
    Poisson clock. Snapshots are taken at construction, every
    ``snapshot_interval_ns`` of simulated time, and at the end of ``run()``.
    """

    def __init__(
        self,
        params: Optional[OrderFlowParams] = None,
        snapshot_interval_ns: int = 100_000_000,  # 0.1s of simulated time
        seed: Optional[int] = None,
    ) -> None:
        self.params = params or OrderFlowParams()
        self.snapshot_interval_ns = snapshot_interval_ns
        self.book = OrderBook()
        self.engine = MatchingEngine(self.book)
        self.simulator = OrderFlowSimulator(
            params=self.params,
            book=self.book,
            rng=random.Random(seed),
        )

        self.trades: List[Trade] = []
        self.snapshots: List[BookSnapshot] = []

        self._n_events_processed = 0
        self._n_new_orders = 0
        self._n_cancels_attempted = 0
        self._n_cancels_succeeded = 0
        self._last_snapshot_time = 0

        # Snapshot the empty book at t=0
        self._snapshot(self.simulator.current_time)

    # ---- public API --------------------------------------------------------

    @property
    def events_processed(self) -> int:
        return self._n_events_processed

    def run(self, n_events: int) -> SimulationStats:
        """Run the simulator for ``n_events`` events. Returns aggregate stats."""
        if n_events < 0:
            raise ValueError("n_events must be non-negative")
        for event in self.simulator.generate(n_events):
            self._handle_event(event)
            self._n_events_processed += 1
            self._maybe_snapshot()
        self._snapshot(self.simulator.current_time)  # final snapshot
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
        """All recorded trades as a DataFrame."""
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
        """All book snapshots as a DataFrame."""
        if not self.snapshots:
            return pd.DataFrame()
        return pd.DataFrame([s.__dict__ for s in self.snapshots])

    # ---- internals ---------------------------------------------------------

    def _handle_event(self, event: FlowEvent) -> None:
        if event.is_cancel:
            self._n_cancels_attempted += 1
            try:
                self.engine.cancel(event.cancel_order_id)
                self._n_cancels_succeeded += 1
            except KeyError:
                pass  # already filled between sampling and processing
            return
        self._n_new_orders += 1
        self.trades.extend(self.engine.submit(event.order))

    def _maybe_snapshot(self) -> None:
        now = self.simulator.current_time
        if now - self._last_snapshot_time >= self.snapshot_interval_ns:
            self._snapshot(now)

    def _snapshot(self, ts: int) -> None:
        bb_level = self.book.best_bid()
        ba_level = self.book.best_ask()
        self.snapshots.append(BookSnapshot(
            timestamp=ts,
            mid=self.book.mid_price,
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