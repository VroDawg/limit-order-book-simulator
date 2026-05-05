"""Tests for the PriceLevel class."""
import pytest

from lob.order import Order, OrderType, Side
from lob.price_level import PriceLevel


def make_order(
    order_id: int,
    qty: int = 100,
    price: float = 50.0,
    timestamp: int = 1_000,
) -> Order:
    """Helper to build a basic limit buy order at ``price``."""
    return Order(
        order_id=order_id,
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        quantity=qty,
        timestamp=timestamp,
        price=price,
    )


class TestPriceLevelCreation:
    def test_create_valid(self) -> None:
        level = PriceLevel(price=50.0)
        assert level.price == 50.0
        assert level.is_empty
        assert level.order_count == 0
        assert level.total_volume == 0

    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValueError, match="price must be positive"):
            PriceLevel(price=-1.0)

    def test_zero_price_rejected(self) -> None:
        with pytest.raises(ValueError, match="price must be positive"):
            PriceLevel(price=0)


class TestPriceLevelAddOrder:
    def test_add_single_order(self) -> None:
        level = PriceLevel(50.0)
        level.add_order(make_order(1, qty=100))
        assert level.order_count == 1
        assert level.total_volume == 100
        assert not level.is_empty

    def test_add_multiple_orders(self) -> None:
        level = PriceLevel(50.0)
        level.add_order(make_order(1, qty=100))
        level.add_order(make_order(2, qty=200))
        level.add_order(make_order(3, qty=50))
        assert level.order_count == 3
        assert level.total_volume == 350

    def test_add_wrong_price_rejected(self) -> None:
        level = PriceLevel(50.0)
        with pytest.raises(ValueError, match="does not match level price"):
            level.add_order(make_order(1, price=51.0))

    def test_add_duplicate_order_id_rejected(self) -> None:
        level = PriceLevel(50.0)
        level.add_order(make_order(1))
        with pytest.raises(ValueError, match="already exists"):
            level.add_order(make_order(1))


class TestPriceLevelTimePriority:
    def test_iteration_in_arrival_order(self) -> None:
        level = PriceLevel(50.0)
        level.add_order(make_order(1, timestamp=1_000))
        level.add_order(make_order(2, timestamp=2_000))
        level.add_order(make_order(3, timestamp=3_000))
        assert [o.order_id for o in level] == [1, 2, 3]

    def test_peek_front_returns_first(self) -> None:
        level = PriceLevel(50.0)
        level.add_order(make_order(1))
        level.add_order(make_order(2))
        assert level.peek_front().order_id == 1

    def test_peek_front_empty_raises(self) -> None:
        level = PriceLevel(50.0)
        with pytest.raises(IndexError, match="empty"):
            level.peek_front()


class TestPriceLevelRemoveOrder:
    def test_remove_existing_order(self) -> None:
        level = PriceLevel(50.0)
        level.add_order(make_order(1, qty=100))
        level.add_order(make_order(2, qty=200))
        removed = level.remove_order(1)
        assert removed.order_id == 1
        assert level.order_count == 1
        assert level.total_volume == 200

    def test_remove_preserves_priority(self) -> None:
        level = PriceLevel(50.0)
        level.add_order(make_order(1))
        level.add_order(make_order(2))
        level.add_order(make_order(3))
        level.remove_order(2)
        assert [o.order_id for o in level] == [1, 3]

    def test_remove_nonexistent_raises(self) -> None:
        level = PriceLevel(50.0)
        level.add_order(make_order(1))
        with pytest.raises(KeyError, match="not found"):
            level.remove_order(99)


class TestPriceLevelFillFront:
    def test_partial_fill(self) -> None:
        level = PriceLevel(50.0)
        level.add_order(make_order(1, qty=100))
        filled = level.fill_front(30)
        assert filled == 30
        assert level.order_count == 1
        assert level.total_volume == 70

    def test_full_fill_removes_order(self) -> None:
        level = PriceLevel(50.0)
        level.add_order(make_order(1, qty=100))
        level.add_order(make_order(2, qty=200))
        filled = level.fill_front(100)
        assert filled == 100
        assert level.order_count == 1
        assert level.peek_front().order_id == 2

    def test_overfill_capped_at_remaining(self) -> None:
        level = PriceLevel(50.0)
        level.add_order(make_order(1, qty=100))
        filled = level.fill_front(150)
        assert filled == 100
        assert level.is_empty

    def test_fill_empty_raises(self) -> None:
        level = PriceLevel(50.0)
        with pytest.raises(IndexError, match="empty"):
            level.fill_front(10)

    def test_fill_zero_qty_raises(self) -> None:
        level = PriceLevel(50.0)
        level.add_order(make_order(1))
        with pytest.raises(ValueError, match="fill_qty must be positive"):
            level.fill_front(0)