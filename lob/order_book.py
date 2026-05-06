"""The two-sided order book: sorted bids and asks built from PriceLevels."""
from __future__ import annotations

from typing import Optional

from sortedcontainers import SortedDict

from lob.order import Order, Side
from lob.price_level import PriceLevel


class OrderBook:
    """Two-sided limit order book.

    Bids and asks are stored as SortedDicts keyed by price. The best bid is the
    highest bid price (``peekitem(-1)``); the best ask is the lowest ask price
    (``peekitem(0)``).

    A flat ``order_id -> (side, price)`` index gives O(1) cancellation.
    Matching is intentionally NOT performed here — the matching engine handles
    crossing orders before they reach the book.
    """

    def __init__(self) -> None:
        self._bids: "SortedDict[float, PriceLevel]" = SortedDict()
        self._asks: "SortedDict[float, PriceLevel]" = SortedDict()
        self._order_locations: dict[int, tuple[Side, float]] = {}

    # ---- mutators ----------------------------------------------------------

    def add_order(self, order: Order) -> None:
        """Rest a limit order on the appropriate side of the book."""
        if order.price is None:
            raise ValueError("market orders cannot rest on the book")
        if order.order_id in self._order_locations:
            raise ValueError(f"order_id {order.order_id} already exists in book")

        side_dict = self._bids if order.side == Side.BUY else self._asks
        if order.price not in side_dict:
            side_dict[order.price] = PriceLevel(order.price)
        side_dict[order.price].add_order(order)
        self._order_locations[order.order_id] = (order.side, order.price)

    def cancel_order(self, order_id: int) -> Order:
        """Cancel an order in the book by id; returns the cancelled order."""
        if order_id not in self._order_locations:
            raise KeyError(f"order_id {order_id} not found in book")
        side, price = self._order_locations.pop(order_id)
        side_dict = self._bids if side == Side.BUY else self._asks
        level = side_dict[price]
        order = level.remove_order(order_id)
        if level.is_empty:
            del side_dict[price]
        order.cancel()
        return order
    def has_order(self, order_id: int) -> bool:
        """Whether an order with this id is currently resting in the book."""
        return order_id in self._order_locations
        
    def consume_from_top(self, side: Side, fill_qty: int) -> tuple[Order, int]:
        """Consume up to ``fill_qty`` from the front order of the best level
        on ``side``. Used by the matching engine when an opposing order matches.

        Returns ``(maker_order, actual_qty_filled)``. Cleans up in place: if
        the maker is fully filled it's removed from the location index, and
        if the level becomes empty it's removed from the side.
        """
        side_dict = self._bids if side == Side.BUY else self._asks
        if not side_dict:
            raise IndexError(f"no orders on {side} side")
        _, level = (
            side_dict.peekitem(-1) if side == Side.BUY else side_dict.peekitem(0)
        )
        front = level.peek_front()
        actual = level.fill_front(fill_qty)
        if front.remaining_quantity == 0:
            self._order_locations.pop(front.order_id, None)
        if level.is_empty:
            del side_dict[level.price]
        return front, actual

    def get_all_order_ids(self) -> list[int]:
        """Return a snapshot list of all currently resting order ids."""
        return list(self._order_locations.keys())

    # ---- top-of-book queries -----------------------------------------------

    def best_bid(self) -> Optional[PriceLevel]:
        """Highest-priced bid level, or None if no bids."""
        if not self._bids:
            return None
        _, level = self._bids.peekitem(-1)
        return level

    def best_ask(self) -> Optional[PriceLevel]:
        """Lowest-priced ask level, or None if no asks."""
        if not self._asks:
            return None
        _, level = self._asks.peekitem(0)
        return level

    @property
    def best_bid_price(self) -> Optional[float]:
        level = self.best_bid()
        return level.price if level else None

    @property
    def best_ask_price(self) -> Optional[float]:
        level = self.best_ask()
        return level.price if level else None

    @property
    def mid_price(self) -> Optional[float]:
        """Average of best bid and best ask. None if either side is empty."""
        bb, ba = self.best_bid_price, self.best_ask_price
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    @property
    def spread(self) -> Optional[float]:
        """Best ask minus best bid. None if either side is empty."""
        bb, ba = self.best_bid_price, self.best_ask_price
        if bb is None or ba is None:
            return None
        return ba - bb

    # ---- depth snapshots ---------------------------------------------------

    def get_bid_levels(self, depth: int = 5) -> list[PriceLevel]:
        """Top ``depth`` bid levels, sorted highest price first."""
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        n = min(depth, len(self._bids))
        return [self._bids.peekitem(-1 - i)[1] for i in range(n)]

    def get_ask_levels(self, depth: int = 5) -> list[PriceLevel]:
        """Top ``depth`` ask levels, sorted lowest price first."""
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        n = min(depth, len(self._asks))
        return [self._asks.peekitem(i)[1] for i in range(n)]

    # ---- size accessors ----------------------------------------------------

    @property
    def bid_count(self) -> int:
        """Number of distinct bid price levels."""
        return len(self._bids)

    @property
    def ask_count(self) -> int:
        """Number of distinct ask price levels."""
        return len(self._asks)

    @property
    def total_orders(self) -> int:
        """Total resting orders across both sides."""
        return len(self._order_locations)

    def __repr__(self) -> str:
        return (
            f"OrderBook(best_bid={self.best_bid_price}, "
            f"best_ask={self.best_ask_price}, "
            f"bid_levels={self.bid_count}, ask_levels={self.ask_count})"
        )
