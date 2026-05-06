"""Smoke tests for plotting — verifies it runs and saves without crashing."""
import os
import tempfile

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless test runs

from lob.plotting import plot_simulation
from lob.simulation import Simulation


def test_plot_runs_and_saves() -> None:
    sim = Simulation(seed=42)
    sim.run(500)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "test.png")
        plot_simulation(sim, save_path=out, show=False)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


def test_plot_handles_empty_simulation() -> None:
    """No trades, no snapshots beyond initial — should still render."""
    sim = Simulation(seed=42)
    sim.run(0)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "empty.png")
        plot_simulation(sim, save_path=out, show=False)
        assert os.path.exists(out)