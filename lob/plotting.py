"""Visualization of Simulation results."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt

from lob.simulation import Simulation


def plot_simulation(
    sim: Simulation,
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """Render a 6-panel summary dashboard of a completed Simulation.

    Returns the matplotlib Figure so callers can further tweak it.
    """
    snaps = sim.snapshots_df()
    trades = sim.trades_df()

    snaps_sec = snaps["timestamp"] / 1e9 if len(snaps) else snaps["timestamp"]
    trades_sec = trades["timestamp"] / 1e9 if len(trades) else trades["timestamp"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    if title is None:
        title = (
            f"LOB Simulation — {sim.events_processed:,} events, "
            f"{len(trades):,} trades, {sim.simulator.current_time / 1e9:.2f}s"
        )
    fig.suptitle(title, fontsize=14)

    # --- 1. Mid price evolution
    ax = axes[0, 0]
    ax.plot(snaps_sec, snaps["mid"], linewidth=1, color="C0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mid price ($)")
    ax.set_title("Mid price evolution")
    ax.grid(True, alpha=0.3)

    # --- 2. Spread over time
    ax = axes[0, 1]
    ax.plot(snaps_sec, snaps["spread"], linewidth=0.6, color="C1", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spread ($)")
    ax.set_title("Bid-ask spread")
    ax.grid(True, alpha=0.3)

    # --- 3. Top-of-book volumes
    ax = axes[0, 2]
    ax.plot(snaps_sec, snaps["bid_volume_l1"],
            label="Bid L1", color="C2", linewidth=0.8, alpha=0.8)
    ax.plot(snaps_sec, snaps["ask_volume_l1"],
            label="Ask L1", color="C3", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Volume at top")
    ax.set_title("Top-of-book volumes")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 4. Trade prints scatter
    ax = axes[1, 0]
    if len(trades):
        buy = trades["aggressor_side"] == "BUY"
        ax.scatter(trades_sec[buy], trades["price"][buy],
                   s=trades["quantity"][buy] * 0.3, alpha=0.4,
                   color="C0", label="Buy aggressor")
        ax.scatter(trades_sec[~buy], trades["price"][~buy],
                   s=trades["quantity"][~buy] * 0.3, alpha=0.4,
                   color="C3", label="Sell aggressor")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trade price ($)")
    ax.set_title("Trade prints (size ∝ qty)")
    ax.grid(True, alpha=0.3)

    # --- 5. Price distribution
    ax = axes[1, 1]
    if len(trades):
        ax.hist(trades["price"], bins=40, color="C4",
                alpha=0.8, edgecolor="white")
        mean_px = trades["price"].mean()
        ax.axvline(mean_px, color="black", linestyle="--", linewidth=1,
                   label=f"mean ${mean_px:.4f}")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Trade price ($)")
    ax.set_ylabel("Trades")
    ax.set_title("Trade price distribution")
    ax.grid(True, alpha=0.3)

    # --- 6. Resting orders
    ax = axes[1, 2]
    ax.plot(snaps_sec, snaps["resting_orders"],
            linewidth=0.8, color="C5")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Resting orders")
    ax.set_title("Total resting orders (book depth)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig