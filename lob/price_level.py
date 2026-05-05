"""A single price level in the order book — FIFO queue at one price."""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterator

from lob.order import Order


class PriceLevel:
    """A FIFO queue of active orders at a single price.

    Orders are filled in time priority: the first order to arrive at this
    price is the first to be filled when an opposing order matches.

    Uses ``OrderedDict`` keyed by order_id, giving O(1) append, peek,
    pop-front, and remove-by-id. This mirrors the intrusive doubly-linked
    list + hashmap design used in production HFT matching engines.
    """

    def __init__(self, price: float) -> None:
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        self.price = price
        self._orders: "OrderedDict[int, Order]" = OrderedDict()
        self._total_volume = 0  # cached sum of remaining_quantity, O(1) access

    @property
    def total_volume(self) -> int:
        """Total remaining quantity across all orders at this level."""
        return self._total_volume

    @property
    def order_count(self) -> int:
        """Number of active orders at this level."""
        return len(self._orders)

    @property
    def is_empty(self) -> bool:
        return len(self._orders) == 0

    def add_order(self, order: Order) -> None:
        """Append an order to the back of the queue (time priority)."""
        if order.price != self.price:
            raise ValueError(
                f"order price {order.price} does not match level price {self.price}"
            )
        if order.order_id in self._orders:
            raise ValueError(f"order_id {order.order_id} already exists at this level")
        self._orders[order.order_id] = order
        self._total_volume += order.remaining_quantity

    def remove_order(self, order_id: int) -> Order:
        """Remove an order by id (e.g. cancellation). Returns the removed order."""
        if order_id not in self._orders:
            raise KeyError(f"order_id {order_id} not found at price {self.price}")
        order = self._orders.pop(order_id)
        self._total_volume -= order.remaining_quantity
        return order

    def peek_front(self) -> Order:
        """Return the front (oldest) order without removing it."""
        if self.is_empty:
            raise IndexError("price level is empty")
        return next(iter(self._orders.values()))

    def fill_front(self, fill_qty: int) -> int:
        """Fill the front order by ``fill_qty``; remove it if fully filled.

        Returns the actual quantity filled (capped by the front order's remaining qty).
        """
        if fill_qty <= 0:
            raise ValueError(f"fill_qty must be positive, got {fill_qty}")
        if self.is_empty:
            raise IndexError("price level is empty")

        front = self.peek_front()
        actual_fill = min(fill_qty, front.remaining_quantity)
        front.fill(actual_fill)
        self._total_volume -= actual_fill

        if front.remaining_quantity == 0:
            self._orders.popitem(last=False)

        return actual_fill

    def __iter__(self) -> Iterator[Order]:
        """Iterate orders in time priority (front to back)."""
        return iter(self._orders.values())

    def __len__(self) -> int:
        return len(self._orders)

    def __repr__(self) -> str:
        return (
            f"PriceLevel(price={self.price}, "
            f"orders={self.order_count}, volume={self.total_volume})"
        )