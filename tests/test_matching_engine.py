"""Tests for the MatchingEngine."""
import pytest

from lob.matching_engine import MatchingEngine, Trade
from lob.order import Order, OrderStatus, OrderType, Side
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


def make_market(
    order_id: int,
    side: Side,
    qty: int = 100,
    timestamp: int = 2_000,
) -> Order:
    return Order(
        order_id=order_id,
        side=side,
        order_type=OrderType.MARKET,
        quantity=qty,
        timestamp=timestamp,
    )


@pytest.fixture
def engine() -> MatchingEngine:
    return MatchingEngine(OrderBook())


class TestNonCrossingLimit:
    def test_limit_buy_below_best_ask_rests(self, engine: MatchingEngine) -> None:
        engine.submit(make_limit(1, Side.SELL, 51.0, qty=100))
        trades = engine.submit(make_limit(2, Side.BUY, 50.0, qty=100))
        assert trades == []
        assert engine.book.best_bid_price == 50.0
        assert engine.book.best_ask_price == 51.0

    def test_limit_sell_above_best_bid_rests(self, engine: MatchingEngine) -> None:
        engine.submit(make_limit(1, Side.BUY, 49.0, qty=100))
        trades = engine.submit(make_limit(2, Side.SELL, 51.0, qty=100))
        assert trades == []
        assert engine.book.best_bid_price == 49.0
        assert engine.book.best_ask_price == 51.0


class TestCrossingLimit:
    def test_full_fill(self, engine: MatchingEngine) -> None:
        engine.submit(make_limit(1, Side.SELL, 51.0, qty=100))
        trades = engine.submit(make_limit(2, Side.BUY, 51.0, qty=100))
        assert len(trades) == 1
        assert trades[0].quantity == 100
        assert trades[0].price == 51.0
        # Both sides empty after match
        assert engine.book.best_ask() is None
        assert engine.book.best_bid() is None

    def test_partial_fill_aggressor_rests_remainder(
        self, engine: MatchingEngine
    ) -> None:
        engine.submit(make_limit(1, Side.SELL, 51.0, qty=60))
        trades = engine.submit(make_limit(2, Side.BUY, 51.0, qty=100))
        assert len(trades) == 1
        assert trades[0].quantity == 60
        assert engine.book.best_ask() is None
        # Aggressor's leftover 40 rests as a bid at 51.0
        assert engine.book.best_bid_price == 51.0
        assert engine.book.best_bid().total_volume == 40

    def test_partial_fill_maker_remains(self, engine: MatchingEngine) -> None:
        engine.submit(make_limit(1, Side.SELL, 51.0, qty=100))
        trades = engine.submit(make_limit(2, Side.BUY, 51.0, qty=40))
        assert len(trades) == 1
        assert trades[0].quantity == 40
        assert engine.book.best_ask().total_volume == 60  # 100 - 40

    def test_walks_multiple_levels(self, engine: MatchingEngine) -> None:
        engine.submit(make_limit(1, Side.SELL, 51.0, qty=50))
        engine.submit(make_limit(2, Side.SELL, 51.5, qty=50))
        engine.submit(make_limit(3, Side.SELL, 52.0, qty=50))
        trades = engine.submit(make_limit(4, Side.BUY, 51.5, qty=120))
        # Buy at 51.5 should match levels at 51.0 and 51.5 only
        assert len(trades) == 2
        assert trades[0].price == 51.0
        assert trades[0].quantity == 50
        assert trades[1].price == 51.5
        assert trades[1].quantity == 50
        # Remaining 20 rests as a bid at 51.5
        assert engine.book.best_bid_price == 51.5
        assert engine.book.best_bid().total_volume == 20
        # Ask at 52.0 untouched
        assert engine.book.best_ask_price == 52.0


class TestMarketOrders:
    def test_market_buy_fills_at_best_ask(self, engine: MatchingEngine) -> None:
        engine.submit(make_limit(1, Side.SELL, 51.0, qty=100))
        trades = engine.submit(make_market(2, Side.BUY, qty=60))
        assert len(trades) == 1
        assert trades[0].price == 51.0
        assert trades[0].quantity == 60

    def test_market_walks_book(self, engine: MatchingEngine) -> None:
        engine.submit(make_limit(1, Side.SELL, 51.0, qty=40))
        engine.submit(make_limit(2, Side.SELL, 52.0, qty=40))
        trades = engine.submit(make_market(3, Side.BUY, qty=70))
        assert len(trades) == 2
        assert [t.price for t in trades] == [51.0, 52.0]
        assert [t.quantity for t in trades] == [40, 30]

    def test_market_exhausts_book_no_rest(self, engine: MatchingEngine) -> None:
        engine.submit(make_limit(1, Side.SELL, 51.0, qty=50))
        market = make_market(2, Side.BUY, qty=200)
        trades = engine.submit(market)
        assert len(trades) == 1
        assert trades[0].quantity == 50
        # Market order doesn't rest; book is now empty on ask side
        assert engine.book.best_ask() is None
        assert engine.book.best_bid() is None  # didn't rest as a bid
        assert market.remaining_quantity == 150  # unfilled


class TestTimePriority:
    def test_oldest_at_same_price_fills_first(
        self, engine: MatchingEngine
    ) -> None:
        engine.submit(make_limit(1, Side.SELL, 51.0, qty=50, timestamp=1_000))
        engine.submit(make_limit(2, Side.SELL, 51.0, qty=50, timestamp=2_000))
        trades = engine.submit(make_limit(3, Side.BUY, 51.0, qty=50))
        assert len(trades) == 1
        assert trades[0].maker_order_id == 1  # the older one


class TestTradeProperties:
    def test_price_is_makers_price(self, engine: MatchingEngine) -> None:
        # Maker rests at 51.0; aggressor crosses at 52.0 — trade should be at 51.0
        engine.submit(make_limit(1, Side.SELL, 51.0, qty=100))
        trades = engine.submit(make_limit(2, Side.BUY, 52.0, qty=100))
        assert trades[0].price == 51.0  # NOT 52.0

    def test_trade_ids_sequential(self, engine: MatchingEngine) -> None:
        engine.submit(make_limit(1, Side.SELL, 51.0, qty=30))
        engine.submit(make_limit(2, Side.SELL, 51.5, qty=30))
        trades = engine.submit(make_limit(3, Side.BUY, 52.0, qty=60))
        assert [t.trade_id for t in trades] == [1, 2]


class TestCancel:
    def test_cancel_resting_order(self, engine: MatchingEngine) -> None:
        engine.submit(make_limit(1, Side.BUY, 49.0, qty=100))
        cancelled = engine.cancel(1)
        assert cancelled.status == OrderStatus.CANCELLED
        assert engine.book.best_bid() is None