"""Drive a Simulation and inspect the captured trades and snapshots.

Run from the project root:
    python -m examples.simulation_demo
"""
from __future__ import annotations

from lob.order_flow import OrderFlowParams
from lob.simulation import Simulation


def main() -> None:
    sim = Simulation(
        params=OrderFlowParams(reference_price=100.0),
        snapshot_interval_ns=100_000_000,  # 0.1s
        seed=42,
    )
    stats = sim.run(n_events=10_000)

    print("=== Simulation stats ===")
    for k, v in stats.__dict__.items():
        if k == "elapsed_ns":
            print(f"  elapsed              {v / 1e9:,.2f} s ({v:,} ns)")
        else:
            print(f"  {k:<20} {v}")

    trades = sim.trades_df()
    print()
    print(f"=== Trades ({len(trades)} total) — first 5 ===")
    print(trades.head().to_string(index=False))
    print()
    print("=== Trade price stats ===")
    if len(trades):
        print(trades["price"].describe().to_string())

    snaps = sim.snapshots_df()
    print()
    print(f"=== Snapshots ({len(snaps)} total) — first 3 ===")
    print(snaps.head(3).to_string(index=False))
    print()
    print(f"=== Snapshots — last 3 ===")
    print(snaps.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()