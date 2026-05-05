"""Tests for the OrderBook class."""
import pytest

from lob.order import Order, OrderType, Side
from lob.order_book import OrderBook


def make_limit(
    order_id: int,
    side: Side,
    price: float,
    qty: int = 100,
    timestamp: int = 1_000,
) -> Order:
    return Order(
        order_id=order_id,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=qty,
        timestamp=timestamp,
        price=price,
    )


class TestEmptyBook:
    def test_empty_state(self) -> None:
        book = OrderBook()
        assert book.best_bid() is None
        assert book.best_ask() is None
        assert book.best_bid_price is None
        assert book.best_ask_price is None
        assert book.mid_price is None
        assert book.spread is None
        assert book.bid_count == 0
        assert book.ask_count == 0
        assert book.total_orders == 0


class TestAddOrder:
    def test_add_single_bid(self) -> None:
        book = OrderBook()
        book.add_order(make_limit(1, Side.BUY, 49.0, qty=100))
        assert book.best_bid_price == 49.0
        assert book.best_bid().total_volume == 100
        assert book.bid_count == 1
        assert book.ask_count == 0

    def test_add_single_ask(self) -> None:
        book = OrderBook()
        book.add_order(make_limit(1, Side.SELL, 51.0, qty=100))
        assert book.best_ask_price == 51.0
        assert book.ask_count == 1
        assert book.bid_count == 0

    def test_best_bid_is_highest(self) -> None:
        book = OrderBook()
        book.add_order(make_limit(1, Side.BUY, 49.0))
        book.add_order(make_limit(2, Side.BUY, 49.5))
        book.add_order(make_limit(3, Side.BUY, 48.0))
        assert book.best_bid_price == 49.5

    def test_best_ask_is_lowest(self) -> None:
        book = OrderBook()
        book.add_order(make_limit(1, Side.SELL, 51.0))
        book.add_order(make_limit(2, Side.SELL, 50.5))
        book.add_order(make_limit(3, Side.SELL, 52.0))
        assert book.best_ask_price == 50.5

    def test_multiple_orders_same_price(self) -> None:
        book = OrderBook()
        book.add_order(make_limit(1, Side.BUY, 49.0, qty=100))
        book.add_order(make_limit(2, Side.BUY, 49.0, qty=200))
        assert book.bid_count == 1
        assert book.best_bid().total_volume == 300
        assert book.best_bid().order_count == 2

    def test_market_order_rejected(self) -> None:
        book = OrderBook()
        market = Order(order_id=1, side=Side.BUY, order_type=OrderType.MARKET,
                       quantity=100, timestamp=1_000)
        with pytest.raises(ValueError, match="market orders cannot rest"):
            book.add_order(market)

    def test_duplicate_order_id_rejected(self) -> None:
        book = OrderBook()
        book.add_order(make_limit(1, Side.BUY, 49.0))
        with pytest.raises(ValueError, match="already exists"):
            book.add_order(make_limit(1, Side.BUY, 50.0))


class TestCancelOrder:
    def test_cancel_existing(self) -> None:
        book = OrderBook()
        book.add_order(make_limit(1, Side.BUY, 49.0))
        order = book.cancel_order(1)
        assert order.order_id == 1
        assert book.bid_count == 0
        assert book.total_orders == 0

    def test_cancel_one_at_busy_level(self) -> None:
        book = OrderBook()
        book.add_order(make_limit(1, Side.BUY, 49.0, qty=100))
        book.add_order(make_limit(2, Side.BUY, 49.0, qty=200))
        book.cancel_order(1)
        assert book.bid_count == 1
        assert book.best_bid().total_volume == 200
        assert book.best_bid().order_count == 1

    def test_cancel_removes_empty_level(self) -> None:
        book = OrderBook()
        book.add_order(make_limit(1, Side.BUY, 49.0))
        book.add_order(make_limit(2, Side.BUY, 50.0))
        book.cancel_order(2)
        assert book.bid_count == 1
        assert book.best_bid_price == 49.0

    def test_cancel_nonexistent_raises(self) -> None:
        book = OrderBook()
        with pytest.raises(KeyError, match="not found"):
            book.cancel_order(99)


class TestSpreadAndMid:
    def test_with_both_sides(self) -> None:
        book = OrderBook()
        book.add_order(make_limit(1, Side.BUY, 49.0))
        book.add_order(make_limit(2, Side.SELL, 51.0))
        assert book.spread == 2.0
        assert book.mid_price == 50.0

    def test_one_side_empty(self) -> None:
        book = OrderBook()
        book.add_order(make_limit(1, Side.BUY, 49.0))
        assert book.spread is None
        assert book.mid_price is None


class TestDepthSnapshot:
    def test_top_n_bids(self) -> None:
        book = OrderBook()
        for i, price in enumerate([48.0, 49.0, 47.0, 49.5, 48.5], start=1):
            book.add_order(make_limit(i, Side.BUY, price))
        levels = book.get_bid_levels(depth=3)
        assert [lvl.price for lvl in levels] == [49.5, 49.0, 48.5]

    def test_top_n_asks(self) -> None:
        book = OrderBook()
        for i, price in enumerate([52.0, 51.0, 53.0, 50.5, 51.5], start=1):
            book.add_order(make_limit(i, Side.SELL, price))
        levels = book.get_ask_levels(depth=3)
        assert [lvl.price for lvl in levels] == [50.5, 51.0, 51.5]

    def test_depth_larger_than_levels(self) -> None:
        book = OrderBook()
        book.add_order(make_limit(1, Side.BUY, 49.0))
        book.add_order(make_limit(2, Side.BUY, 49.5))
        levels = book.get_bid_levels(depth=10)
        assert len(levels) == 2

    def test_depth_zero_rejected(self) -> None:
        book = OrderBook()
        with pytest.raises(ValueError, match="depth must be positive"):
            book.get_bid_levels(depth=0)