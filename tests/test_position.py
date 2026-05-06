"""Tests for Position."""
import pytest

from lob.order import Side
from lob.position import Position


class TestPosition:
    def test_initial_state(self) -> None:
        p = Position()
        assert p.inventory == 0
        assert p.cash == 0.0
        assert p.mtm(100.0) == 0.0

    def test_buy_increases_inventory_decreases_cash(self) -> None:
        p = Position()
        p.apply_fill(Side.BUY, price=100.0, qty=50)
        assert p.inventory == 50
        assert p.cash == -5_000.0

    def test_sell_decreases_inventory_increases_cash(self) -> None:
        p = Position()
        p.apply_fill(Side.SELL, price=100.0, qty=30)
        assert p.inventory == -30
        assert p.cash == 3_000.0

    def test_round_trip_zero_pnl(self) -> None:
        p = Position()
        p.apply_fill(Side.BUY, 100.0, 50)
        p.apply_fill(Side.SELL, 100.0, 50)
        assert p.inventory == 0
        assert p.mtm(100.0) == 0.0

    def test_round_trip_with_profit(self) -> None:
        p = Position()
        p.apply_fill(Side.BUY, 100.0, 50)
        p.apply_fill(Side.SELL, 101.0, 50)
        assert p.inventory == 0
        assert p.cash == 50.0
        assert p.mtm(100.0) == 50.0

    def test_mtm_open_long(self) -> None:
        p = Position()
        p.apply_fill(Side.BUY, 100.0, 50)
        assert p.mtm(101.0) == 50.0   # cash -5000 + 50*101 = 50

    def test_mtm_open_short(self) -> None:
        p = Position()
        p.apply_fill(Side.SELL, 100.0, 30)
        assert p.mtm(99.0) == 30.0    # cash 3000 + (-30)*99 = 30

    def test_negative_qty_rejected(self) -> None:
        p = Position()
        with pytest.raises(ValueError, match="qty must be positive"):
            p.apply_fill(Side.BUY, 100.0, 0)

    def test_negative_price_rejected(self) -> None:
        p = Position()
        with pytest.raises(ValueError, match="price must be positive"):
            p.apply_fill(Side.BUY, -1.0, 10)