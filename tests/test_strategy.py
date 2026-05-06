"""Tests for Strategy and FixedSpreadMarketMaker (isolation, no Simulation)."""
import pytest

from lob.matching_engine import MatchingEngine
from lob.order import Order, OrderType, Side
from lob.order_book import OrderBook
from lob.strategy import FixedSpreadMarketMaker


@pytest.fixture
def setup():
    book = OrderBook()
    engine = MatchingEngine(book)
    return book, engine


def populate(book: OrderBook) -> None:
    """Add reference orders so mid is well-defined at $100."""
    book.add_order(Order(
        order_id=1, side=Side.BUY, order_type=OrderType.LIMIT,
        quantity=100, timestamp=0, price=99.0,
    ))
    book.add_order(Order(
        order_id=2, side=Side.SELL, order_type=OrderType.LIMIT,
        quantity=100, timestamp=0, price=101.0,
    ))


class TestFixedSpreadInit:
    def test_no_quotes_on_empty_book(self, setup) -> None:
        book, engine = setup
        mm = FixedSpreadMarketMaker(engine, book)
        mm.on_event(trades=[], current_time=1000)
        assert mm.bid_order_id is None
        assert mm.ask_order_id is None
        assert len(mm.active_orders) == 0


class TestFixedSpreadQuoting:
    def test_places_two_quotes_with_mid(self, setup) -> None:
        book, engine = setup
        populate(book)
        mm = FixedSpreadMarketMaker(engine, book, half_spread_ticks=2, quote_size=50)
        mm.on_event(trades=[], current_time=1000)
        assert mm.bid_order_id is not None
        assert mm.ask_order_id is not None
        assert len(mm.active_orders) == 2

    def test_quotes_at_correct_offsets(self, setup) -> None:
        book, engine = setup
        populate(book)
        mm = FixedSpreadMarketMaker(engine, book, half_spread_ticks=5, tick_size=0.01)
        mm.on_event(trades=[], current_time=1000)
        bid = mm.active_orders[mm.bid_order_id]
        ask = mm.active_orders[mm.ask_order_id]
        assert bid.price == pytest.approx(99.95)
        assert ask.price == pytest.approx(100.05)

    def test_does_not_requote_if_mid_stable(self, setup) -> None:
        book, engine = setup
        populate(book)
        mm = FixedSpreadMarketMaker(engine, book)
        mm.on_event(trades=[], current_time=1000)
        first_bid, first_ask = mm.bid_order_id, mm.ask_order_id
        mm.on_event(trades=[], current_time=2000)
        assert mm.bid_order_id == first_bid
        assert mm.ask_order_id == first_ask

    def test_requotes_when_mid_moves(self, setup) -> None:
        book, engine = setup
        populate(book)
        mm = FixedSpreadMarketMaker(engine, book, tick_size=0.01)
        mm.on_event(trades=[], current_time=1000)
        first_bid = mm.bid_order_id
        # Tighten the ask side so the mid drops
        book.add_order(Order(
            order_id=99, side=Side.SELL, order_type=OrderType.LIMIT,
            quantity=100, timestamp=0, price=100.00,
        ))
        mm.on_event(trades=[], current_time=2000)
        assert mm.bid_order_id != first_bid


class TestFixedSpreadFills:
    def test_position_updates_when_filled(self, setup) -> None:
        book, engine = setup
        populate(book)
        mm = FixedSpreadMarketMaker(engine, book, half_spread_ticks=2, quote_size=50)
        mm.on_event(trades=[], current_time=1000)
        # MM bid is at 99.98. An aggressive market sell should hit it.
        seller = Order(
            order_id=999, side=Side.SELL, order_type=OrderType.MARKET,
            quantity=30, timestamp=2000,
        )
        trades = engine.submit(seller)
        mm.on_event(trades=trades, current_time=2000)
        assert mm.position.inventory == 30
        assert mm.position.cash < 0  # paid for the buy
        assert len(mm.fills) == 1


class TestStrategyCancelAll:
    def test_cancel_all(self, setup) -> None:
        book, engine = setup
        populate(book)
        mm = FixedSpreadMarketMaker(engine, book)
        mm.on_event(trades=[], current_time=1000)
        assert len(mm.active_orders) == 2
        n = mm.cancel_all()
        assert n == 2
        assert len(mm.active_orders) == 0