"""Tests for the metrics module."""
import math

import pytest

from lob.metrics import StrategyMetrics, compute_metrics
from lob.simulation import Simulation
from lob.strategy import FixedSpreadMarketMaker


def _sim_with_mm() -> Simulation:
    sim = Simulation(
        seed=42,
        strategy_factory=lambda e, b: FixedSpreadMarketMaker(
            e, b, half_spread_ticks=2, quote_size=20,
        ),
    )
    sim.run(2_000)
    return sim


class TestComputeMetrics:
    def test_returns_metrics_object(self) -> None:
        m = compute_metrics(_sim_with_mm())
        assert isinstance(m, StrategyMetrics)

    def test_no_strategy_raises(self) -> None:
        sim = Simulation(seed=42)
        sim.run(100)
        with pytest.raises(ValueError, match="no strategy"):
            compute_metrics(sim)

    def test_finite_values(self) -> None:
        m = compute_metrics(_sim_with_mm())
        for field in ("final_pnl", "sharpe_per_snapshot", "max_drawdown",
                      "inventory_std", "avg_fill_price",
                      "avg_adverse_selection", "avg_realized_spread"):
            val = getattr(m, field)
            assert math.isfinite(val), f"{field} is not finite: {val}"

    def test_max_drawdown_non_positive(self) -> None:
        m = compute_metrics(_sim_with_mm())
        assert m.max_drawdown <= 0

    def test_inventory_metrics_non_negative(self) -> None:
        m = compute_metrics(_sim_with_mm())
        assert m.inventory_max >= 0
        assert m.inventory_std >= 0

    def test_fill_rate_in_unit_interval(self) -> None:
        m = compute_metrics(_sim_with_mm())
        assert 0.0 <= m.fill_rate <= 1.0

    def test_n_fills_matches_strategy(self) -> None:
        sim = _sim_with_mm()
        m = compute_metrics(sim)
        assert m.n_fills == len(sim.strategy.fills)