"""Tests for the OrderFlowSimulator."""
import random

import pytest

from lob.order import Order, OrderType, Side
from lob.order_book import OrderBook
from lob.order_flow import FlowEvent, OrderFlowParams, OrderFlowSimulator


def make_sim(seed: int = 42, **param_overrides) -> OrderFlowSimulator:
    book = OrderBook()
    params = OrderFlowParams(**param_overrides)
    return OrderFlowSimulator(params=params, book=book, rng=random.Random(seed))


def orders_only(events):
    return [e.order for e in events if e.is_new_order]


class TestBasicGeneration:
    def test_generates_n_events(self) -> None:
        sim = make_sim(lambda_cancel=0.0)  # no cancels for clean count
        events = list(sim.generate(50))
        assert len(events) == 50
        assert all(isinstance(e, FlowEvent) for e in events)

    def test_order_ids_sequential(self) -> None:
        sim = OrderFlowSimulator(
            OrderFlowParams(lambda_cancel=0.0),
            OrderBook(),
            rng=random.Random(1),
            starting_order_id=100,
        )
        events = list(sim.generate(20))
        ids = [e.order.order_id for e in events if e.is_new_order]
        assert ids == list(range(100, 120))

    def test_timestamps_monotonic(self) -> None:
        sim = make_sim()
        timestamps = [e.timestamp for e in sim.generate(100)]
        assert all(t2 >= t1 for t1, t2 in zip(timestamps, timestamps[1:]))

    def test_sizes_within_bounds(self) -> None:
        sim = make_sim(mean_size=20.0, max_size=200, lambda_cancel=0.0)
        events = list(sim.generate(500))
        sizes = [e.order.quantity for e in events if e.is_new_order]
        assert all(1 <= s <= 200 for s in sizes)

    def test_deterministic_with_seed(self) -> None:
        sim_a = make_sim(seed=123, lambda_cancel=0.0)
        sim_b = make_sim(seed=123, lambda_cancel=0.0)
        a = [(e.order.order_id, e.timestamp, e.order.quantity, e.order.price)
             for e in sim_a.generate(50)]
        b = [(e.order.order_id, e.timestamp, e.order.quantity, e.order.price)
             for e in sim_b.generate(50)]
        assert a == b


class TestEventTypeMix:
    def test_proportions_match_rates(self) -> None:
        sim = make_sim(seed=7, lambda_cancel=0.0)
        events = list(sim.generate(2_000))
        n_limits = sum(1 for e in events
                       if e.is_new_order and e.order.order_type == OrderType.LIMIT)
        n_markets = sum(1 for e in events
                        if e.is_new_order and e.order.order_type == OrderType.MARKET)
        ratio = n_limits / max(n_markets, 1)
        assert 3.0 < ratio < 8.0

    def test_buy_sell_balanced(self) -> None:
        sim = make_sim(seed=11, lambda_cancel=0.0)
        events = list(sim.generate(1_000))
        n_buy = sum(1 for e in events if e.is_new_order and e.order.side == Side.BUY)
        n_sell = sum(1 for e in events if e.is_new_order and e.order.side == Side.SELL)
        assert abs(n_buy - n_sell) < 100


class TestLimitPlacement:
    def test_limit_buy_below_or_at_best_ask(self) -> None:
        book = OrderBook()
        book.add_order(Order(
            order_id=1, side=Side.SELL, order_type=OrderType.LIMIT,
            quantity=100, timestamp=0, price=100.0,
        ))
        sim = OrderFlowSimulator(
            params=OrderFlowParams(
                lambda_limit_sell=0.0, lambda_market_buy=0.0,
                lambda_market_sell=0.0, lambda_cancel=0.0,
            ),
            book=book, rng=random.Random(3), starting_order_id=2,
        )
        for event in sim.generate(50):
            assert event.is_new_order
            o = event.order
            assert o.side == Side.BUY
            assert o.order_type == OrderType.LIMIT
            assert o.price <= book.best_ask_price

    def test_limit_sell_above_or_at_best_bid(self) -> None:
        book = OrderBook()
        book.add_order(Order(
            order_id=1, side=Side.BUY, order_type=OrderType.LIMIT,
            quantity=100, timestamp=0, price=100.0,
        ))
        sim = OrderFlowSimulator(
            params=OrderFlowParams(
                lambda_limit_buy=0.0, lambda_market_buy=0.0,
                lambda_market_sell=0.0, lambda_cancel=0.0,
            ),
            book=book, rng=random.Random(3), starting_order_id=2,
        )
        for event in sim.generate(50):
            assert event.is_new_order
            o = event.order
            assert o.side == Side.SELL
            assert o.order_type == OrderType.LIMIT
            assert o.price >= book.best_bid_price


class TestEdgeCases:
    def test_empty_book_no_crash(self) -> None:
        sim = make_sim(lambda_cancel=0.0)
        events = list(sim.generate(50))
        assert len(events) == 50

    def test_market_orders_have_no_price(self) -> None:
        sim = make_sim(seed=99, lambda_cancel=0.0)
        market_orders = [
            e.order for e in sim.generate(500)
            if e.is_new_order and e.order.order_type == OrderType.MARKET
        ]
        assert len(market_orders) > 0
        assert all(o.price is None for o in market_orders)

    def test_zero_n_events(self) -> None:
        sim = make_sim()
        assert list(sim.generate(0)) == []

    def test_negative_n_events_rejected(self) -> None:
        sim = make_sim()
        with pytest.raises(ValueError, match="non-negative"):
            list(sim.generate(-1))

    def test_all_zero_rates_rejected(self) -> None:
        sim = make_sim()
        sim.params.lambda_limit_buy = 0
        sim.params.lambda_limit_sell = 0
        sim.params.lambda_market_buy = 0
        sim.params.lambda_market_sell = 0
        sim.params.lambda_cancel = 0
        with pytest.raises(ValueError, match="positive"):
            list(sim.generate(1))


class TestCancellations:
    def test_cancels_target_resting_orders(self) -> None:
        # Pre-load book; only allow cancels; verify they target real ids
        book = OrderBook()
        for i in range(1, 11):
            book.add_order(Order(
                order_id=i, side=Side.BUY, order_type=OrderType.LIMIT,
                quantity=50, timestamp=0, price=99.0 + i * 0.01,
            ))
        sim = OrderFlowSimulator(
            params=OrderFlowParams(
                lambda_limit_buy=0.0, lambda_limit_sell=0.0,
                lambda_market_buy=0.0, lambda_market_sell=0.0,
                lambda_cancel=1e-7,
            ),
            book=book, rng=random.Random(13), starting_order_id=11,
        )
        events = list(sim.generate(20))
        assert all(e.is_cancel for e in events)
        assert all(e.cancel_order_id in range(1, 11) for e in events)

    def test_cancels_skipped_when_book_empty(self) -> None:
        # All-cancel rates with empty book: simulator should still produce
        # n events by re-sampling (but it can never get a non-cancel here),
        # so the loop would be infinite. Guard by adding tiny order rates.
        sim = make_sim(
            seed=5,
            lambda_limit_buy=1e-9, lambda_limit_sell=1e-9,
            lambda_market_buy=0.0, lambda_market_sell=0.0,
            lambda_cancel=1e-7,
        )
        events = list(sim.generate(30))
        # Initially the book is empty, so first events must be new orders
        assert events[0].is_new_order

    def test_cancel_event_no_order_field(self) -> None:
        book = OrderBook()
        book.add_order(Order(
            order_id=1, side=Side.BUY, order_type=OrderType.LIMIT,
            quantity=50, timestamp=0, price=99.0,
        ))
        sim = OrderFlowSimulator(
            params=OrderFlowParams(
                lambda_limit_buy=0.0, lambda_limit_sell=0.0,
                lambda_market_buy=0.0, lambda_market_sell=0.0,
                lambda_cancel=1e-7,
            ),
            book=book, rng=random.Random(0), starting_order_id=2,
        )
        event = next(sim.generate(1))
        assert event.is_cancel
        assert not event.is_new_order
        assert event.order is None
        assert event.cancel_order_id == 1