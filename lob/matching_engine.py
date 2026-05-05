"""Price-time priority matching engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from lob.order import Order, OrderType, Side
from lob.order_book import OrderBook


@dataclass
class Trade:
    """A trade resulting from a match between an aggressor and a maker.

    By convention, the trade prints at the *maker's* price (the resting
    order's price), not the aggressor's. This is standard price-time priority.
    """
    trade_id: int
    aggressor_order_id: int
    maker_order_id: int
    aggressor_side: Side
    price: float
    quantity: int
    timestamp: int


class MatchingEngine:
    """Price-time priority matching engine.

    Incoming orders are first matched against the opposing side of the book.
    Any unfilled remainder of a *limit* order rests on the book at its
    limit price. *Market* orders never rest — if they exhaust the book,
    they simply stop.
    """

    def __init__(self, book: OrderBook) -> None:
        self.book = book
        self._next_trade_id = 1

    def submit(self, order: Order) -> List[Trade]:
        """Submit an order. Returns any trades it generated."""
        if order.order_type == OrderType.LIMIT:
            return self._handle_limit(order)
        if order.order_type == OrderType.MARKET:
            return self._handle_market(order)
        raise ValueError(f"unsupported order type: {order.order_type}")

    def cancel(self, order_id: int) -> Order:
        """Cancel a resting order. Returns the cancelled order."""
        return self.book.cancel_order(order_id)

    # ---- internals ---------------------------------------------------------

    def _handle_limit(self, order: Order) -> List[Trade]:
        trades = self._match(order)
        if order.is_active and order.remaining_quantity > 0:
            self.book.add_order(order)
        return trades

    def _handle_market(self, order: Order) -> List[Trade]:
        return self._match(order)

    def _match(self, aggressor: Order) -> List[Trade]:
        """Match the aggressor against the opposing side until done or exhausted."""
        trades: List[Trade] = []
        opposing_side = Side.SELL if aggressor.side == Side.BUY else Side.BUY

        while aggressor.remaining_quantity > 0:
            best_price = self._best_opposing_price(aggressor.side)
            if best_price is None or not self._crosses(aggressor, best_price):
                break
            maker, fill_qty = self.book.consume_from_top(
                opposing_side, aggressor.remaining_quantity
            )
            aggressor.fill(fill_qty)
            trades.append(
                Trade(
                    trade_id=self._next_trade_id,
                    aggressor_order_id=aggressor.order_id,
                    maker_order_id=maker.order_id,
                    aggressor_side=aggressor.side,
                    price=best_price,
                    quantity=fill_qty,
                    timestamp=aggressor.timestamp,
                )
            )
            self._next_trade_id += 1
        return trades

    def _best_opposing_price(self, aggressor_side: Side):
        """Return the best opposing price, or None if that side is empty."""
        if aggressor_side == Side.BUY:
            return self.book.best_ask_price
        return self.book.best_bid_price

    def _crosses(self, aggressor: Order, opposing_price: float) -> bool:
        """Whether the aggressor crosses an opposing level at ``opposing_price``."""
        if aggressor.order_type == OrderType.MARKET:
            return True
        if aggressor.side == Side.BUY:
            return aggressor.price >= opposing_price
        return aggressor.price <= opposing_price