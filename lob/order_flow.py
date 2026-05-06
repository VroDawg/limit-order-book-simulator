"""Stochastic order flow simulator with Poisson arrivals and cancellations."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

from lob.order import Order, OrderType, Side
from lob.order_book import OrderBook
from typing import Callable, Iterator, Optional

class EventType(Enum):
    """The five arrival processes the simulator drives."""
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL"
    CANCEL = "CANCEL"


@dataclass
class OrderFlowParams:
    """All knobs for the flow simulator. Rates are events per nanosecond."""
    lambda_limit_buy: float = 1e-7
    lambda_limit_sell: float = 1e-7
    lambda_market_buy: float = 2e-8
    lambda_market_sell: float = 2e-8
    # Cancels are typically the highest-rate event type in real flow.
    lambda_cancel: float = 1.5e-7

    mean_price_offset_ticks: float = 3.0
    tick_size: float = 0.01

    mean_size: float = 50.0
    max_size: int = 1_000

    reference_price: float = 100.0


@dataclass
class FlowEvent:
    """A simulator event: either a new order to submit or a cancel target."""
    timestamp: int
    order: Optional[Order] = None
    cancel_order_id: Optional[int] = None

    @property
    def is_new_order(self) -> bool:
        return self.order is not None

    @property
    def is_cancel(self) -> bool:
        return self.cancel_order_id is not None


class OrderFlowSimulator:
    """Generates a stream of FlowEvents from independent Poisson processes.

    Cancel events sample a target order_id uniformly at random from the
    book's currently resting orders. If the book has no resting orders when
    a cancel is sampled, the event is treated as a phantom (time advances,
    no event emitted) and the simulator re-samples.
    """

    def __init__(
        self,
        params: OrderFlowParams,
        book: OrderBook,
        rng: Optional[random.Random] = None,
        starting_timestamp: int = 0,
        starting_order_id: int = 1,
        is_protected: Optional["Callable[[int], bool]"] = None,
    ) -> None:
        self.params = params
        self.book = book
        self.rng = rng or random.Random()
        self._timestamp = starting_timestamp
        self._next_order_id = starting_order_id
        self.is_protected = is_protected or (lambda _oid: False)

    @property
    def current_time(self) -> int:
        return self._timestamp

    def generate(self, n_events: int) -> Iterator[FlowEvent]:
        """Yield ``n_events`` FlowEvents, advancing the internal clock."""
        if n_events < 0:
            raise ValueError("n_events must be non-negative")
        for _ in range(n_events):
            yield self._next_event()

    # ---- internals ---------------------------------------------------------

    def _next_event(self) -> FlowEvent:
        rates = [
            self.params.lambda_limit_buy,
            self.params.lambda_limit_sell,
            self.params.lambda_market_buy,
            self.params.lambda_market_sell,
            self.params.lambda_cancel,
        ]
        total = sum(rates)
        if total <= 0:
            raise ValueError("at least one event rate must be positive")

        # Loop in case we sample a CANCEL with an empty book (phantom event).
        while True:
            dt = self.rng.expovariate(total)
            self._timestamp += int(dt)

            u = self.rng.random() * total
            cumulative = 0.0
            event_type = EventType.LIMIT_BUY
            for et, rate in zip(EventType, rates):
                cumulative += rate
                if u <= cumulative:
                    event_type = et
                    break

            if event_type == EventType.CANCEL:
                ids = [
                    oid for oid in self.book.get_all_order_ids()
                    if not self.is_protected(oid)
                ]
                if not ids:
                    continue  # phantom: time advanced, no event
                target = self.rng.choice(ids)
                return FlowEvent(timestamp=self._timestamp, cancel_order_id=target)
            
            return FlowEvent(
                timestamp=self._timestamp,
                order=self._build_order(event_type),
            )

    def _build_order(self, event_type: EventType) -> Order:
        order_id = self._next_order_id
        self._next_order_id += 1
        size = self._draw_size()
        ts = self._timestamp

        if event_type == EventType.MARKET_BUY:
            return Order(order_id=order_id, side=Side.BUY,
                         order_type=OrderType.MARKET, quantity=size, timestamp=ts)
        if event_type == EventType.MARKET_SELL:
            return Order(order_id=order_id, side=Side.SELL,
                         order_type=OrderType.MARKET, quantity=size, timestamp=ts)

        side = Side.BUY if event_type == EventType.LIMIT_BUY else Side.SELL
        price = self._draw_limit_price(side)
        return Order(order_id=order_id, side=side, order_type=OrderType.LIMIT,
                     quantity=size, timestamp=ts, price=price)

    def _draw_size(self) -> int:
        mean = self.params.mean_size
        if mean <= 1:
            return 1
        p = 1.0 / mean
        u = self.rng.random()
        if u <= 0.0:
            return 1
        size = max(1, int(math.ceil(math.log(u) / math.log(1.0 - p))))
        return min(size, self.params.max_size)

    def _draw_limit_price(self, side: Side) -> float:
        tick = self.params.tick_size
        mean_offset = self.params.mean_price_offset_ticks
        offset = self.rng.expovariate(1.0 / mean_offset) * tick

        if side == Side.BUY:
            ref = self.book.best_ask_price
        else:
            ref = self.book.best_bid_price
        if ref is None:
            ref = self.book.mid_price
        if ref is None:
            ref = self.params.reference_price

        price = ref - offset if side == Side.BUY else ref + offset
        price = round(round(price / tick) * tick, 8)
        return max(price, tick)