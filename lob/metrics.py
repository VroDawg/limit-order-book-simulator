"""Performance metrics for backtested strategies."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import pandas as pd

from lob.order import Side
from lob.simulation import Simulation
from lob.strategy import StrategyFill


@dataclass
class StrategyMetrics:
    """Standard market-making performance metrics."""
    final_pnl: float
    sharpe_per_snapshot: float
    max_drawdown: float
    inventory_max: int
    inventory_std: float
    n_fills: int
    total_volume: int
    avg_fill_price: float
    fill_rate: float
    avg_adverse_selection: float  # $/share; positive = bad
    avg_realized_spread: float    # $/share; positive = good


def compute_metrics(
    sim: Simulation,
    adverse_horizon_ns: int = 1_000_000_000,  # 1 simulated second
) -> StrategyMetrics:
    """Compute the full metric set for a backtest."""
    if sim.strategy is None:
        raise ValueError("Simulation has no strategy")

    strat = sim.strategy
    snap_df = sim.strategy_snapshots_df()
    book_df = sim.snapshots_df()

    final_pnl = float(snap_df["pnl"].iloc[-1]) if len(snap_df) else 0.0

    pnl_changes = snap_df["pnl"].diff().dropna()
    if len(pnl_changes) > 1 and pnl_changes.std() > 0:
        sharpe = float(pnl_changes.mean() / pnl_changes.std())
    else:
        sharpe = 0.0

    running_max = snap_df["pnl"].cummax()
    drawdown = snap_df["pnl"] - running_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    inv = snap_df["inventory"]
    inv_max = int(inv.abs().max()) if len(inv) else 0
    inv_std = float(inv.std()) if len(inv) > 1 else 0.0

    n_fills = len(strat.fills)
    total_volume = sum(f.quantity for f in strat.fills)
    avg_fill_price = (
        sum(f.price * f.quantity for f in strat.fills) / total_volume
        if total_volume > 0 else 0.0
    )
    total_market_trades = len(sim.trades)
    fill_rate = n_fills / total_market_trades if total_market_trades > 0 else 0.0

    avg_adv = _avg_adverse_selection(strat.fills, book_df, adverse_horizon_ns)
    avg_real_spread = _avg_realized_spread(strat.fills, book_df)

    return StrategyMetrics(
        final_pnl=final_pnl,
        sharpe_per_snapshot=sharpe,
        max_drawdown=max_dd,
        inventory_max=inv_max,
        inventory_std=inv_std,
        n_fills=n_fills,
        total_volume=total_volume,
        avg_fill_price=avg_fill_price,
        fill_rate=fill_rate,
        avg_adverse_selection=avg_adv,
        avg_realized_spread=avg_real_spread,
    )


def _avg_adverse_selection(
    fills: List[StrategyFill],
    book_df: pd.DataFrame,
    horizon_ns: int,
) -> float:
    """Average per-share adverse selection cost.

    For a buy at price P with mid M_future after ``horizon``, cost = P − M_future.
    For a sell, cost = M_future − P. Positive = filled by informed flow.
    """
    if not fills or len(book_df) == 0:
        return 0.0
    mids = book_df[["timestamp", "mid"]].dropna()
    if len(mids) == 0:
        return 0.0

    timestamps = mids["timestamp"].values
    mid_values = mids["mid"].values

    total_cost = 0.0
    total_qty = 0
    for fill in fills:
        idx = timestamps.searchsorted(fill.timestamp + horizon_ns)
        if idx >= len(timestamps):
            continue
        future_mid = mid_values[idx]
        if pd.isna(future_mid):
            continue
        cost = (fill.price - future_mid) if fill.side == Side.BUY else (future_mid - fill.price)
        total_cost += cost * fill.quantity
        total_qty += fill.quantity
    return total_cost / total_qty if total_qty > 0 else 0.0


def _avg_realized_spread(
    fills: List[StrategyFill],
    book_df: pd.DataFrame,
) -> float:
    """Average per-share realized spread captured at fill time.

    For a buy at price P with mid M at fill time, spread = M − P (captured below mid).
    For a sell, spread = P − M.
    """
    if not fills or len(book_df) == 0:
        return 0.0
    mids = book_df[["timestamp", "mid"]].dropna()
    if len(mids) == 0:
        return 0.0

    timestamps = mids["timestamp"].values
    mid_values = mids["mid"].values

    total_spread = 0.0
    total_qty = 0
    for fill in fills:
        idx = timestamps.searchsorted(fill.timestamp)
        if idx > 0:
            idx -= 1  # use snapshot at or just before fill
        if idx >= len(mid_values):
            continue
        mid_at_fill = mid_values[idx]
        if pd.isna(mid_at_fill):
            continue
        spread = (mid_at_fill - fill.price) if fill.side == Side.BUY else (fill.price - mid_at_fill)
        total_spread += spread * fill.quantity
        total_qty += fill.quantity
    return total_spread / total_qty if total_qty > 0 else 0.0