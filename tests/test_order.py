"""Tests for the Order class."""
import pytest

from lob.order import Order, OrderStatus, OrderType, Side


class TestOrderCreation:
    """Construction and validation."""

    def test_create_valid_limit_buy(self) -> None:
        order = Order(
            order_id=1,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            timestamp=1_000,
            price=50.0,
        )
        assert order.order_id == 1
        assert order.side == Side.BUY
        assert order.quantity == 100
        assert order.price == 50.0
        assert order.filled_quantity == 0
        assert order.status == OrderStatus.NEW

    def test_create_valid_market_sell(self) -> None:
        order = Order(
            order_id=2,
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=50,
            timestamp=2_000,
        )
        assert order.price is None
        assert order.status == OrderStatus.NEW

    def test_zero_quantity_rejected(self) -> None:
        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(order_id=1, side=Side.BUY, order_type=OrderType.LIMIT,
                  quantity=0, timestamp=1_000, price=50.0)

    def test_negative_quantity_rejected(self) -> None:
        with pytest.raises(ValueError, match="quantity must be positive"):
            Order(order_id=1, side=Side.BUY, order_type=OrderType.LIMIT,
                  quantity=-10, timestamp=1_000, price=50.0)

    def test_limit_without_price_rejected(self) -> None:
        with pytest.raises(ValueError, match="limit order requires a price"):
            Order(order_id=1, side=Side.BUY, order_type=OrderType.LIMIT,
                  quantity=100, timestamp=1_000)

    def test_market_with_price_rejected(self) -> None:
        with pytest.raises(ValueError, match="market order must not have a price"):
            Order(order_id=1, side=Side.BUY, order_type=OrderType.MARKET,
                  quantity=100, timestamp=1_000, price=50.0)

    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValueError, match="price must be positive"):
            Order(order_id=1, side=Side.BUY, order_type=OrderType.LIMIT,
                  quantity=100, timestamp=1_000, price=-1.0)


class TestOrderFill:
    """Fill logic."""

    def _make_order(self, qty: int = 100) -> Order:
        return Order(order_id=1, side=Side.BUY, order_type=OrderType.LIMIT,
                     quantity=qty, timestamp=1_000, price=50.0)

    def test_partial_fill(self) -> None:
        order = self._make_order(100)
        order.fill(30)
        assert order.filled_quantity == 30
        assert order.remaining_quantity == 70
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.is_active

    def test_full_fill(self) -> None:
        order = self._make_order(100)
        order.fill(100)
        assert order.filled_quantity == 100
        assert order.remaining_quantity == 0
        assert order.status == OrderStatus.FILLED
        assert not order.is_active

    def test_two_partial_fills_complete(self) -> None:
        order = self._make_order(100)
        order.fill(40)
        order.fill(60)
        assert order.status == OrderStatus.FILLED
        assert order.remaining_quantity == 0

    def test_overfill_rejected(self) -> None:
        order = self._make_order(100)
        order.fill(50)
        with pytest.raises(ValueError, match="exceeds remaining"):
            order.fill(60)

    def test_zero_fill_rejected(self) -> None:
        order = self._make_order(100)
        with pytest.raises(ValueError, match="fill quantity must be positive"):
            order.fill(0)


class TestOrderCancel:
    """Cancellation logic."""

    def _make_order(self) -> Order:
        return Order(order_id=1, side=Side.BUY, order_type=OrderType.LIMIT,
                     quantity=100, timestamp=1_000, price=50.0)

    def test_cancel_new_order(self) -> None:
        order = self._make_order()
        order.cancel()
        assert order.status == OrderStatus.CANCELLED
        assert not order.is_active

    def test_cancel_partially_filled(self) -> None:
        order = self._make_order()
        order.fill(30)
        order.cancel()
        assert order.status == OrderStatus.CANCELLED

    def test_cannot_cancel_filled(self) -> None:
        order = self._make_order()
        order.fill(100)
        with pytest.raises(ValueError, match="cannot cancel"):
            order.cancel()

    def test_cannot_cancel_twice(self) -> None:
        order = self._make_order()
        order.cancel()
        with pytest.raises(ValueError, match="cannot cancel"):
            order.cancel()