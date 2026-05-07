"""Compare strategies across two market regimes: balanced vs. bearish.

The point: in a benign mean-reverting market, simple strategies win; in a
trending market, the inventory-aware strategies should outperform.

Run from project root:
    python -m examples.regime_comparison
"""
from __future__ import annotations

from typing import Dict, Tuple

from lob.metrics import StrategyMetrics, compute_metrics
from lob.order_flow import OrderFlowParams
from lob.simulation import Simulation
from lob.strategy import (
    AvellanedaStoikovMarketMaker,
    FixedSpreadMarketMaker,
    InventoryAwareMarketMaker,
)

N_EVENTS = 20_000
SEED = 42

REGIMES: Dict[str, OrderFlowParams] = {
    "Balanced": OrderFlowParams(
        reference_price=100.0,
        lambda_market_buy=2e-8,
        lambda_market_sell=2e-8,
    ),
    "Bearish": OrderFlowParams(
        reference_price=100.0,
        lambda_market_buy=1e-8,    # half the usual
        lambda_market_sell=5e-8,   # 2.5x the usual
    ),
}

STRATEGIES = {
    "FixedSpread": lambda e, b: FixedSpreadMarketMaker(
        e, b, half_spread_ticks=2, quote_size=20, tick_size=0.01,
    ),
    "InvAware": lambda e, b: InventoryAwareMarketMaker(
        e, b, half_spread_ticks=2, quote_size=20, tick_size=0.01,
        skew_per_share=0.0001,
    ),
    "AvelStoikov": lambda e, b: AvellanedaStoikovMarketMaker(
        e, b,
        sigma=0.1, gamma=0.05, kappa=50.0, horizon_seconds=1.0,
        quote_size=20, tick_size=0.01,
    ),
}


def run(factory, params) -> Tuple[Simulation, StrategyMetrics]:
    sim = Simulation(
        params=params,
        strategy_factory=factory,
        snapshot_interval_ns=50_000_000,
        seed=SEED,
    )
    sim.run(N_EVENTS)
    return sim, compute_metrics(sim)


def print_regime_table(regime_name: str, regime_params: OrderFlowParams,
                       results: Dict[str, Tuple[Simulation, StrategyMetrics]]) -> None:
    final_mid = next(iter(results.values()))[0].book.mid_price
    drift = ((final_mid or regime_params.reference_price)
             - regime_params.reference_price)
    print(f"\n=== Regime: {regime_name} "
          f"(start ${regime_params.reference_price:.2f}, "
          f"end ${final_mid:.4f}, drift ${drift:+.4f}) ===")

    headers = list(results.keys())
    print(f"{'Metric':<22} " + "".join(f"{h:>14}" for h in headers))
    print("-" * (22 + 14 * len(headers)))

    rows = [
        ("Final P&L",         "final_pnl",            "${:>10,.2f}"),
        ("Sharpe (per snap)", "sharpe_per_snapshot",  "{:>13.4f}"),
        ("Max drawdown",      "max_drawdown",         "${:>10,.2f}"),
        ("Max abs inventory", "inventory_max",        "{:>13,}"),
        ("Inventory std",     "inventory_std",        "{:>13.2f}"),
        ("Total fills",       "n_fills",              "{:>13,}"),
        ("Realized spread",   "avg_realized_spread",  "${:>10.5f}"),
        ("Adverse selection", "avg_adverse_selection","${:>10.5f}"),
    ]
    for label, attr, fmt in rows:
        line = f"{label:<22} "
        for name in headers:
            _, m = results[name]
            val = getattr(m, attr)
            line += fmt.format(val).rjust(14)
        print(line)


def main() -> None:
    all_results: Dict[str, Dict[str, StrategyMetrics]] = {}

    for regime_name, regime_params in REGIMES.items():
        results = {}
        for strat_name, factory in STRATEGIES.items():
            sim, metrics = run(factory, regime_params)
            results[strat_name] = (sim, metrics)
        print_regime_table(regime_name, regime_params, results)
        all_results[regime_name] = {n: r[1] for n, r in results.items()}

    # P&L summary across regimes — the headline of this exercise
    print("\n\n=== Final P&L across regimes ===")
    print(f"{'Strategy':<14} " + "".join(f"{r:>14}" for r in REGIMES))
    print("-" * (14 + 14 * len(REGIMES)))
    for strat in STRATEGIES:
        line = f"{strat:<14} "
        for regime in REGIMES:
            pnl = all_results[regime][strat].final_pnl
            line += f"${pnl:>11,.2f}".rjust(14)
        print(line)

    # Rank changes — the punchline
    print("\n=== Strategy ranking by P&L per regime ===")
    for regime in REGIMES:
        ranked = sorted(
            STRATEGIES.keys(),
            key=lambda s: all_results[regime][s].final_pnl,
            reverse=True,
        )
        print(f"  {regime:<14} {'  >  '.join(ranked)}")


if __name__ == "__main__":
    main()