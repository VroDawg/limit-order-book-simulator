"""Backtest a FixedSpreadMarketMaker and compute full performance metrics.

Run from project root:
    python -m examples.backtest_demo
"""
from __future__ import annotations

import os

from lob.metrics import compute_metrics
from lob.order_flow import OrderFlowParams
from lob.plotting import plot_backtest
from lob.simulation import Simulation
from lob.strategy import FixedSpreadMarketMaker

OUTPUT_DIR = "plots"
OUTPUT_FILE = "backtest_fixed_spread.png"


def main() -> None:
    sim = Simulation(
        params=OrderFlowParams(reference_price=100.0),
        strategy_factory=lambda engine, book: FixedSpreadMarketMaker(
            engine=engine, book=book,
            half_spread_ticks=2, quote_size=20, tick_size=0.01,
        ),
        snapshot_interval_ns=50_000_000,
        seed=42,
    )
    sim.run(n_events=20_000)

    m = compute_metrics(sim)

    print("=== FixedSpreadMM — performance metrics ===")
    print(f"  Final P&L:               ${m.final_pnl:>10,.2f}")
    print(f"  Sharpe (per snapshot):   {m.sharpe_per_snapshot:>10.4f}")
    print(f"  Max drawdown:            ${m.max_drawdown:>10,.2f}")
    print(f"  Max abs inventory:       {m.inventory_max:>10,}")
    print(f"  Inventory std dev:       {m.inventory_std:>10,.2f}")
    print(f"  Total fills:             {m.n_fills:>10,}")
    print(f"  Total volume:            {m.total_volume:>10,}")
    print(f"  Fill rate:               {m.fill_rate:>10.2%}")
    print(f"  Avg fill price:          ${m.avg_fill_price:>10.4f}")
    print(f"  Avg realized spread:     ${m.avg_realized_spread:>10.5f}/share")
    print(f"  Avg adverse selection:   ${m.avg_adverse_selection:>10.5f}/share")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plot_backtest(sim, save_path=out_path, show=True,
                  title="FixedSpreadMarketMaker — backtest")
    print(f"\nSaved dashboard to {out_path}")


if __name__ == "__main__":
    main()