"""End-to-end smoke test: random order flow through the matching engine.

Run from the project root:
    python examples/demo.py
"""
from __future__ import annotations

import random
from collections import Counter

from lob.matching_engine import MatchingEngine
from lob.order import Order, OrderType, Side
from lob.order_book import OrderBook

FAIR_VALUE = 50.0
N_ORDERS = 200


def random_order(order_id: int, timestamp: int) -> Order:
    """Generate a random order around the fair value."""
    side = random.choice([Side.BUY, Side.SELL])
    order_type = random.choices(
        [OrderType.LIMIT, OrderType.MARKET],
        weights=[0.85, 0.15],  # ~85% limits, 15% markets — closer to real flow
    )[0]
    qty = random.choice([10, 25, 50, 100, 200])

    if order_type == OrderType.MARKET:
        return Order(
            order_id=order_id,
            side=side,
            order_type=OrderType.MARKET,
            quantity=qty,
            timestamp=timestamp,
        )

    # Limit order: place near fair value with some random offset
    if side == Side.BUY:
        price = round(FAIR_VALUE - random.uniform(0.0, 1.0), 2)
    else:
        price = round(FAIR_VALUE + random.uniform(0.0, 1.0), 2)

    return Order(
        order_id=order_id,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=qty,
        timestamp=timestamp,
        price=price,
    )


def print_top_of_book(book: OrderBook, depth: int = 5) -> None:
    """Pretty-print the top ``depth`` levels of each side."""
    bids = book.get_bid_levels(depth)
    asks = book.get_ask_levels(depth)
    print(f"{'BIDS':>22} | {'ASKS':<22}")
    print("-" * 47)
    for i in range(max(len(bids), len(asks))):
        bid_str = (
            f"{bids[i].total_volume:>5} @ {bids[i].price:>8.2f}"
            if i < len(bids) else ""
        )
        ask_str = (
            f"{asks[i].price:<8.2f} @ {asks[i].total_volume:<5}"
            if i < len(asks) else ""
        )
        print(f"{bid_str:>22} | {ask_str:<22}")


def main() -> None:
    random.seed(42)  # deterministic for reproducible demo output
    book = OrderBook()
    engine = MatchingEngine(book)

    all_trades = []
    for i in range(1, N_ORDERS + 1):
        order = random_order(order_id=i, timestamp=i * 1_000)
        trades = engine.submit(order)
        all_trades.extend(trades)

    print("=== Simulation complete ===")
    print(f"Orders submitted:   {N_ORDERS}")
    print(f"Trades generated:   {len(all_trades)}")
    if all_trades:
        total_volume = sum(t.quantity for t in all_trades)
        avg_price = sum(t.price * t.quantity for t in all_trades) / total_volume
        sides = Counter(t.aggressor_side.value for t in all_trades)
        print(f"Total volume:       {total_volume}")
        print(f"VWAP:               ${avg_price:.4f}")
        print(f"Aggressor sides:    {dict(sides)}")

    print()
    print("=== Final book snapshot ===")
    print(f"Best bid:       {book.best_bid_price}")
    print(f"Best ask:       {book.best_ask_price}")
    print(f"Mid:            {book.mid_price}")
    print(f"Spread:         {book.spread}")
    print(f"Bid levels:     {book.bid_count}")
    print(f"Ask levels:     {book.ask_count}")
    print(f"Resting orders: {book.total_orders}")

    print()
    print("=== Top of book ===")
    print_top_of_book(book, depth=5)


if __name__ == "__main__":
    main()
    