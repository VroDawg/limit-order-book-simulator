"""Position tracking: inventory and cash, with mark-to-market P&L."""
from __future__ import annotations

from dataclasses import dataclass

from lob.order import Side


@dataclass
class Position:
    """Tracks inventory (signed share count) and cash for a strategy.

    Mark-to-market P&L at any mark price is ``cash + inventory * mark``.
    This is the standard backtesting formulation: every fill flows cash
    in or out, and open inventory is valued at the current mark.
    """
    inventory: int = 0
    cash: float = 0.0

    def apply_fill(self, side: Side, price: float, qty: int) -> None:
        """Update from a fill. ``side`` is the strategy's side: BUY = bought."""
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if side == Side.BUY:
            self.inventory += qty
            self.cash -= qty * price
        else:
            self.inventory -= qty
            self.cash += qty * price

    def mtm(self, mark_price: float) -> float:
        """Mark-to-market P&L: cash + inventory * mark_price."""
        return self.cash + self.inventory * mark_price

    def __repr__(self) -> str:
        return f"Position(inv={self.inventory}, cash=${self.cash:.2f})"