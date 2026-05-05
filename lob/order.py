"""Order representation for the limit order book."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Side(Enum):
    """Buy (bid) or sell (ask) side of the book."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Type of order. Limit orders rest on the book; market orders execute immediately."""
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(Enum):
    """Lifecycle status of an order."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    """A single order submitted to the book.

    Limit orders specify a price; market orders have ``price=None`` and
    execute against the best available counterparty quotes.
    """
    order_id: int
    side: Side
    order_type: OrderType
    quantity: int
    timestamp: int  # nanoseconds since epoch
    price: Optional[float] = None
    filled_quantity: int = 0
    status: OrderStatus = OrderStatus.NEW

    def __post_init__(self) -> None:
        """Validate the order on creation."""
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("limit order requires a price")
        if self.order_type == OrderType.MARKET and self.price is not None:
            raise ValueError("market order must not have a price")
        if self.price is not None and self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")

    @property
    def remaining_quantity(self) -> int:
        """Quantity not yet filled."""
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        """Whether the order can still be matched."""
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)

    def fill(self, fill_qty: int) -> None:
        """Apply a fill of the given quantity to this order."""
        if fill_qty <= 0:
            raise ValueError(f"fill quantity must be positive, got {fill_qty}")
        if fill_qty > self.remaining_quantity:
            raise ValueError(
                f"fill quantity {fill_qty} exceeds remaining {self.remaining_quantity}"
            )
        self.filled_quantity += fill_qty
        if self.filled_quantity == self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self) -> None:
        """Cancel an active order."""
        if not self.is_active:
            raise ValueError(f"cannot cancel order in status {self.status.value}")
        self.status = OrderStatus.CANCELLED