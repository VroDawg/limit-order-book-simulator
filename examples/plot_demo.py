"""Run a simulation and save a 6-panel dashboard PNG.

Run from project root:
    python -m examples.plot_demo
"""
from __future__ import annotations

import os

from lob.order_flow import OrderFlowParams
from lob.plotting import plot_simulation
from lob.simulation import Simulation

OUTPUT_DIR = "plots"
OUTPUT_FILE = "simulation.png"


def main() -> None:
    sim = Simulation(
        params=OrderFlowParams(reference_price=100.0),
        snapshot_interval_ns=50_000_000,  # 0.05s — finer resolution
        seed=42,
    )
    sim.run(n_events=20_000)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plot_simulation(sim, save_path=out_path, show=True)
    print(f"Saved dashboard to {out_path}")


if __name__ == "__main__":
    main()