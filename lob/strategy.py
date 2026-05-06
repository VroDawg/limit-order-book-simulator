"""Strategy framework: abstract base + first concrete market maker."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from lob.matching_engine import MatchingEngine, Trade
from lob.order import Order, OrderType, Side
from lob.order_book import OrderBook
from lob.position import Position
from dataclasses import dataclass

@dataclass
class StrategyFill:
    """Record of a fill on one of the strategy's own orders."""
    timestamp: int
    price: float
    quantity: int
    side: Side       # the strategy's side: BUY = bought, SELL = sold
    order_id: int

class Strategy(ABC):
    """Abstract strategy: participates in a simulation, tracks its own P&L.

    The simulation calls ``on_event`` after each market event, passing the
    trades that event generated and the current simulated time. The base
    class auto-detects fills against this strategy's own orders and updates
    Position. Subclasses implement ``on_step`` for trading logic.
    """

    def __init__(
        self,
        engine: MatchingEngine,
        book: OrderBook,
        starting_order_id: int = 1_000_000,
    ) -> None:
        self.engine = engine
        self.book = book
        self.position = Position()
        self.active_orders: Dict[int, Order] = {}
        self.fills: List[StrategyFill] = []
        self._next_order_id = starting_order_id
        self._current_time = 0

    # ---- public API --------------------------------------------------------

    def on_event(self, trades: List[Trade], current_time: int) -> None:
        """Hook called by the simulation after each market event."""
        self._current_time = current_time
        for trade in trades:
            self._maybe_apply_fill(trade)
        self._reconcile_with_book()
        self.on_step()

    @abstractmethod
    def on_step(self) -> None:
        """Strategy-specific trading logic, called once per market event."""
        ...

    def submit(
        self,
        side: Side,
        price: Optional[float],
        quantity: int,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Order:
        """Submit an order via the engine and track it. Returns the Order.

        Trades from immediate matches are applied to position automatically.
        """
        order_id = self._next_order_id
        self._next_order_id += 1
        order = Order(
            order_id=order_id,
            side=side,
            order_type=order_type,
            quantity=quantity,
            timestamp=self._current_time,
            price=price,
        )
        self.active_orders[order_id] = order
        for trade in self.engine.submit(order):
            self._maybe_apply_fill(trade)
        return order

    def cancel(self, order_id: int) -> bool:
        """Cancel one of our orders. Returns True if successfully cancelled."""
        if order_id not in self.active_orders:
            return False
        try:
            self.engine.cancel(order_id)
        except KeyError:
            self.active_orders.pop(order_id, None)
            return False
        self.active_orders.pop(order_id, None)
        return True

    def cancel_all(self) -> int:
        """Cancel all active orders. Returns the count cancelled."""
        return sum(1 for oid in list(self.active_orders.keys()) if self.cancel(oid))

    # ---- internals ---------------------------------------------------------

    def _reconcile_with_book(self) -> None:
        """Drop active_orders entries that are no longer in the book.

        Handles the case where one of our orders was cancelled or fully
        filled by a path other than our own ``submit``/``cancel``.
        """
        for order_id in list(self.active_orders.keys()):
            if not self.book.has_order(order_id):
                self.active_orders.pop(order_id, None)
    
    def _maybe_apply_fill(self, trade: Trade) -> None:
        """If a trade involves one of our orders, update Position and record fill."""
        for order_id in (trade.maker_order_id, trade.aggressor_order_id):
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                self.position.apply_fill(order.side, trade.price, trade.quantity)
                self.fills.append(StrategyFill(
                    timestamp=trade.timestamp,
                    price=trade.price,
                    quantity=trade.quantity,
                    side=order.side,
                    order_id=order_id,
                ))
                if order.remaining_quantity == 0:
                    self.active_orders.pop(order_id, None)
                return
                


class FixedSpreadMarketMaker(Strategy):
    """Quotes one bid + one ask at fixed offsets from mid.

    Re-quotes when:
      - either quote is filled (no longer in active_orders), or
      - mid drifts by ≥ ½ tick from when we last quoted.
    """

    def __init__(
        self,
        engine: MatchingEngine,
        book: OrderBook,
        half_spread_ticks: int = 2,
        quote_size: int = 50,
        tick_size: float = 0.01,
        starting_order_id: int = 1_000_000,
    ) -> None:
        super().__init__(engine, book, starting_order_id)
        self.half_spread_ticks = half_spread_ticks
        self.quote_size = quote_size
        self.tick_size = tick_size
        self.bid_order_id: Optional[int] = None
        self.ask_order_id: Optional[int] = None
        self._last_quoted_mid: Optional[float] = None

    def on_step(self) -> None:
        mid = self.book.mid_price
        if mid is None:
            return  # no two-sided market — can't quote

        # Drop dead handles (filled or otherwise gone)
        if self.bid_order_id is not None and self.bid_order_id not in self.active_orders:
            self.bid_order_id = None
        if self.ask_order_id is not None and self.ask_order_id not in self.active_orders:
            self.ask_order_id = None

        if not self._should_requote(mid):
            return

        self._cancel_existing()
        self._place_quotes(mid)
        self._last_quoted_mid = mid

    def _should_requote(self, mid: float) -> bool:
        if self.bid_order_id is None or self.ask_order_id is None:
            return True
        if self._last_quoted_mid is None:
            return True
        return abs(mid - self._last_quoted_mid) >= 0.5 * self.tick_size

    def _cancel_existing(self) -> None:
        if self.bid_order_id is not None:
            self.cancel(self.bid_order_id)
            self.bid_order_id = None
        if self.ask_order_id is not None:
            self.cancel(self.ask_order_id)
            self.ask_order_id = None

    def _place_quotes(self, mid: float) -> None:
        offset = self.half_spread_ticks * self.tick_size
        bid_price = round(round((mid - offset) / self.tick_size) * self.tick_size, 8)
        ask_price = round(round((mid + offset) / self.tick_size) * self.tick_size, 8)
        if bid_price >= ask_price:
            ask_price = round(bid_price + self.tick_size, 8)

        bid = self.submit(Side.BUY, bid_price, self.quote_size)
        ask = self.submit(Side.SELL, ask_price, self.quote_size)
        self.bid_order_id = bid.order_id if bid.is_active else None
        self.ask_order_id = ask.order_id if ask.is_active else None