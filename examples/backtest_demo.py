"""Compare FixedSpread, InventoryAware, and Avellaneda-Stoikov market makers.

Run from project root:
    python -m examples.backtest_demo
"""
from __future__ import annotations

import os

from lob.metrics import StrategyMetrics, compute_metrics
from lob.order_flow import OrderFlowParams
from lob.plotting import plot_backtest
from lob.simulation import Simulation
from lob.strategy import (
    AvellanedaStoikovMarketMaker,
    FixedSpreadMarketMaker,
    InventoryAwareMarketMaker,
)

OUTPUT_DIR = "plots"
N_EVENTS = 20_000
SEED = 42


def run_strategy(factory) -> tuple[Simulation, StrategyMetrics]:
    sim = Simulation(
        params=OrderFlowParams(reference_price=100.0),
        strategy_factory=factory,
        snapshot_interval_ns=50_000_000,
        seed=SEED,
    )
    sim.run(n_events=N_EVENTS)
    return sim, compute_metrics(sim)


def main() -> None:
    sim_fixed, m_fixed = run_strategy(
        lambda e, b: FixedSpreadMarketMaker(
            e, b, half_spread_ticks=2, quote_size=20, tick_size=0.01,
        )
    )
    sim_aware, m_aware = run_strategy(
        lambda e, b: InventoryAwareMarketMaker(
            e, b, half_spread_ticks=2, quote_size=20, tick_size=0.01,
            skew_per_share=0.0001,
        )
    )
    sim_as, m_as = run_strategy(
        lambda e, b: AvellanedaStoikovMarketMaker(
            e, b,
            sigma=0.1, gamma=0.05, kappa=50.0, horizon_seconds=1.0,
            quote_size=20, tick_size=0.01,
        )
    )

    rows = [
        ("Final P&L",         f"${m_fixed.final_pnl:>9,.2f}", f"${m_aware.final_pnl:>9,.2f}", f"${m_as.final_pnl:>9,.2f}"),
        ("Sharpe (per snap)", f"{m_fixed.sharpe_per_snapshot:>10.4f}", f"{m_aware.sharpe_per_snapshot:>10.4f}", f"{m_as.sharpe_per_snapshot:>10.4f}"),
        ("Max drawdown",      f"${m_fixed.max_drawdown:>9,.2f}", f"${m_aware.max_drawdown:>9,.2f}", f"${m_as.max_drawdown:>9,.2f}"),
        ("Max abs inventory", f"{m_fixed.inventory_max:>10,}", f"{m_aware.inventory_max:>10,}", f"{m_as.inventory_max:>10,}"),
        ("Inventory std",     f"{m_fixed.inventory_std:>10.2f}", f"{m_aware.inventory_std:>10.2f}", f"{m_as.inventory_std:>10.2f}"),
        ("Total fills",       f"{m_fixed.n_fills:>10,}", f"{m_aware.n_fills:>10,}", f"{m_as.n_fills:>10,}"),
        ("Fill rate",         f"{m_fixed.fill_rate:>10.2%}", f"{m_aware.fill_rate:>10.2%}", f"{m_as.fill_rate:>10.2%}"),
        ("Realized spread",   f"${m_fixed.avg_realized_spread:>9.5f}", f"${m_aware.avg_realized_spread:>9.5f}", f"${m_as.avg_realized_spread:>9.5f}"),
        ("Adverse selection", f"${m_fixed.avg_adverse_selection:>9.5f}", f"${m_aware.avg_adverse_selection:>9.5f}", f"${m_as.avg_adverse_selection:>9.5f}"),
    ]

    print(f"=== Strategy comparison ({N_EVENTS:,} events, seed={SEED}) ===")
    print(f"{'Metric':<22} {'FixedSpread':>12} {'InvAware':>12} {'AvelStoikov':>12}")
    print("-" * 64)
    for label, a, b, c in rows:
        print(f"{label:<22} {a:>12} {b:>12} {c:>12}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_backtest(sim_fixed, save_path=os.path.join(OUTPUT_DIR, "backtest_fixed_spread.png"),
                  show=False, title="FixedSpreadMM")
    plot_backtest(sim_aware, save_path=os.path.join(OUTPUT_DIR, "backtest_inventory_aware.png"),
                  show=False, title="InventoryAwareMM")
    plot_backtest(sim_as, save_path=os.path.join(OUTPUT_DIR, "backtest_avellaneda_stoikov.png"),
                  show=True, title="AvellanedaStoikovMM")
    print(f"\nSaved 3 dashboards to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()