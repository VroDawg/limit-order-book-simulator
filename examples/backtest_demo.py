"""Compare FixedSpreadMarketMaker vs InventoryAwareMarketMaker.

Run from project root:
    python -m examples.backtest_demo
"""
from __future__ import annotations

import os

from lob.metrics import StrategyMetrics, compute_metrics
from lob.order_flow import OrderFlowParams
from lob.plotting import plot_backtest
from lob.simulation import Simulation
from lob.strategy import FixedSpreadMarketMaker, InventoryAwareMarketMaker

OUTPUT_DIR = "plots"
N_EVENTS = 20_000
SEED = 42


def run_strategy(name: str, factory) -> tuple[Simulation, StrategyMetrics]:
    sim = Simulation(
        params=OrderFlowParams(reference_price=100.0),
        strategy_factory=factory,
        snapshot_interval_ns=50_000_000,
        seed=SEED,
    )
    sim.run(n_events=N_EVENTS)
    return sim, compute_metrics(sim)


def fmt_money(x: float) -> str:
    return f"${x:>10,.2f}"


def fmt_share(x: float) -> str:
    return f"${x:>10.5f}"


def main() -> None:
    sim_fixed, m_fixed = run_strategy(
        "FixedSpread",
        lambda e, b: FixedSpreadMarketMaker(
            e, b, half_spread_ticks=2, quote_size=20, tick_size=0.01,
        ),
    )
    sim_aware, m_aware = run_strategy(
        "InventoryAware",
        lambda e, b: InventoryAwareMarketMaker(
            e, b, half_spread_ticks=2, quote_size=20, tick_size=0.01,
            skew_per_share=0.0001,
        ),
    )

    rows = [
        ("Final P&L",            fmt_money(m_fixed.final_pnl),         fmt_money(m_aware.final_pnl)),
        ("Sharpe (per snap)",    f"{m_fixed.sharpe_per_snapshot:>11.4f}", f"{m_aware.sharpe_per_snapshot:>11.4f}"),
        ("Max drawdown",         fmt_money(m_fixed.max_drawdown),      fmt_money(m_aware.max_drawdown)),
        ("Max abs inventory",    f"{m_fixed.inventory_max:>11,}",      f"{m_aware.inventory_max:>11,}"),
        ("Inventory std",        f"{m_fixed.inventory_std:>11.2f}",    f"{m_aware.inventory_std:>11.2f}"),
        ("Total fills",          f"{m_fixed.n_fills:>11,}",            f"{m_aware.n_fills:>11,}"),
        ("Fill rate",            f"{m_fixed.fill_rate:>11.2%}",        f"{m_aware.fill_rate:>11.2%}"),
        ("Realized spread",      fmt_share(m_fixed.avg_realized_spread), fmt_share(m_aware.avg_realized_spread)),
        ("Adverse selection",    fmt_share(m_fixed.avg_adverse_selection), fmt_share(m_aware.avg_adverse_selection)),
    ]

    print(f"=== Strategy comparison ({N_EVENTS:,} events, seed={SEED}) ===")
    print(f"{'Metric':<22} {'FixedSpread':>14} {'InventoryAware':>16}")
    print("-" * 56)
    for label, a, b in rows:
        print(f"{label:<22} {a:>14} {b:>16}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_backtest(
        sim_fixed,
        save_path=os.path.join(OUTPUT_DIR, "backtest_fixed_spread.png"),
        show=False,
        title="FixedSpreadMM",
    )
    plot_backtest(
        sim_aware,
        save_path=os.path.join(OUTPUT_DIR, "backtest_inventory_aware.png"),
        show=True,
        title="InventoryAwareMM",
    )
    print(f"\nSaved both dashboards to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()