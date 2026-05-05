"""Realistic order flow demo with Poisson arrivals + cancellations.

Run from project root:
    python -m examples.flow_demo
"""
from __future__ import annotations

import random
from collections import Counter

from lob.matching_engine import MatchingEngine
from lob.order_book import OrderBook
from lob.order_flow import OrderFlowParams, OrderFlowSimulator


def main() -> None:
    book = OrderBook()
    engine = MatchingEngine(book)
    params = OrderFlowParams(reference_price=100.0)
    sim = OrderFlowSimulator(params=params, book=book, rng=random.Random(42))

    n_events = 5_000
    all_trades = []
    new_order_types: Counter[str] = Counter()
    n_cancels = 0
    n_failed_cancels = 0

    for event in sim.generate(n_events):
        if event.is_cancel:
            try:
                engine.cancel(event.cancel_order_id)
                n_cancels += 1
            except KeyError:
                # Order may have been fully filled between sampling and processing
                n_failed_cancels += 1
            continue

        order = event.order
        new_order_types[f"{order.side.value}_{order.order_type.value}"] += 1
        all_trades.extend(engine.submit(order))

    elapsed_s = sim.current_time / 1e9

    print("=== Flow simulation summary ===")
    print(f"Events generated:    {n_events}")
    print(f"Simulated time:      {elapsed_s:,.2f} s")
    print(f"Trades:              {len(all_trades)}")
    print(f"Cancellations:       {n_cancels}")
    print(f"Failed cancels:      {n_failed_cancels}  (order already filled)")
    if all_trades:
        vol = sum(t.quantity for t in all_trades)
        vwap = sum(t.price * t.quantity for t in all_trades) / vol
        print(f"Volume traded:       {vol:,}")
        print(f"VWAP:                ${vwap:.4f}")
    print()
    print("New-order mix:")
    for k in sorted(new_order_types):
        print(f"  {k:<25} {new_order_types[k]}")

    print()
    print("=== Final book state ===")
    print(f"Best bid:        {book.best_bid_price}")
    print(f"Best ask:        {book.best_ask_price}")
    print(f"Mid:             {book.mid_price}")
    print(f"Spread:          {book.spread}")
    print(f"Resting orders:  {book.total_orders}")


if __name__ == "__main__":
    main()