"""Backtest a FixedSpreadMarketMaker against simulated noise flow.

Run from project root:
    python -m examples.backtest_demo
"""
from __future__ import annotations

from lob.order_flow import OrderFlowParams
from lob.simulation import Simulation
from lob.strategy import FixedSpreadMarketMaker


def main() -> None:
    sim = Simulation(
        params=OrderFlowParams(reference_price=100.0),
        strategy_factory=lambda engine, book: FixedSpreadMarketMaker(
            engine=engine,
            book=book,
            half_spread_ticks=2,
            quote_size=20,
            tick_size=0.01,
        ),
        snapshot_interval_ns=50_000_000,
        seed=42,
    )
    stats = sim.run(n_events=20_000)

    print("=== Simulation stats ===")
    for k, v in stats.__dict__.items():
        if k == "elapsed_ns":
            print(f"  elapsed              {v / 1e9:,.2f} s")
        else:
            print(f"  {k:<20} {v:,}" if isinstance(v, int) else f"  {k:<20} {v}")

    print()
    strat = sim.strategy
    pos = strat.position
    final_mid = sim.book.mid_price or sim.params.reference_price
    print("=== Strategy results (FixedSpreadMM) ===")
    print(f"  Total fills:         {len(strat.fills):,}")
    print(f"  Final inventory:     {pos.inventory}")
    print(f"  Final cash:          ${pos.cash:,.2f}")
    print(f"  Final mark price:    ${final_mid:.4f}")
    print(f"  Final MTM P&L:       ${pos.mtm(final_mid):,.2f}")
    if strat.fills:
        vwap = (
            sum(t.price * t.quantity for t in strat.fills)
            / sum(t.quantity for t in strat.fills)
        )
        print(f"  Avg fill price:      ${vwap:.4f}")

    snaps = sim.strategy_snapshots_df()
    if len(snaps):
        print()
        print("=== P&L over time (first 3, last 3) ===")
        cols = ["timestamp", "inventory", "cash", "pnl", "n_fills"]
        print(snaps[cols].head(3).to_string(index=False))
        print("...")
        print(snaps[cols].tail(3).to_string(index=False))


if __name__ == "__main__":
    main()