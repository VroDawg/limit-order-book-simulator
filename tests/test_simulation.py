"""Tests for the Simulation runner."""
import pytest

from lob.simulation import BookSnapshot, Simulation, SimulationStats


class TestBasicRun:
    def test_run_zero_events(self) -> None:
        sim = Simulation(seed=42)
        stats = sim.run(0)
        assert stats.events_processed == 0
        assert len(sim.trades) == 0

    def test_negative_n_rejected(self) -> None:
        sim = Simulation(seed=42)
        with pytest.raises(ValueError, match="non-negative"):
            sim.run(-1)

    def test_run_processes_n_events(self) -> None:
        sim = Simulation(seed=42)
        stats = sim.run(100)
        assert stats.events_processed == 100
        assert sim.events_processed == 100

    def test_runs_accumulate(self) -> None:
        sim = Simulation(seed=42)
        sim.run(50)
        sim.run(50)
        assert sim.events_processed == 100

    def test_deterministic_with_seed(self) -> None:
        a, b = Simulation(seed=99), Simulation(seed=99)
        a.run(500)
        b.run(500)
        assert len(a.trades) == len(b.trades)
        if a.trades:
            assert a.trades[0].price == b.trades[0].price
            assert a.trades[-1].timestamp == b.trades[-1].timestamp


class TestSnapshots:
    def test_initial_snapshot_at_construction(self) -> None:
        sim = Simulation(seed=42)
        assert len(sim.snapshots) == 1
        assert sim.snapshots[0].timestamp == 0
        assert sim.snapshots[0].mid is None  # empty book

    def test_snapshots_grow_over_run(self) -> None:
        sim = Simulation(seed=42, snapshot_interval_ns=10_000_000)
        sim.run(2_000)
        assert len(sim.snapshots) > 5

    def test_snapshot_timestamps_monotonic(self) -> None:
        sim = Simulation(seed=42)
        sim.run(500)
        ts = [s.timestamp for s in sim.snapshots]
        assert ts == sorted(ts)

    def test_snapshot_fields_populated_after_activity(self) -> None:
        sim = Simulation(seed=42)
        sim.run(2_000)
        # Find any snapshot with both sides populated
        non_empty = [s for s in sim.snapshots if s.mid is not None]
        assert len(non_empty) > 0
        s = non_empty[0]
        assert s.best_bid is not None
        assert s.best_ask is not None
        assert s.spread is not None and s.spread >= 0


class TestStatsAndDataFrames:
    def test_stats_internal_consistency(self) -> None:
        sim = Simulation(seed=42)
        stats = sim.run(1_000)
        assert stats.trades == len(sim.trades)
        assert stats.cancels_succeeded <= stats.cancels_attempted
        assert stats.new_orders + stats.cancels_attempted == stats.events_processed
        assert stats.total_volume == sum(t.quantity for t in sim.trades)

    def test_trades_df_columns_and_rows(self) -> None:
        sim = Simulation(seed=42)
        sim.run(1_000)
        df = sim.trades_df()
        for col in ["trade_id", "timestamp", "price", "quantity",
                    "aggressor_side", "aggressor_order_id", "maker_order_id"]:
            assert col in df.columns
        assert len(df) == len(sim.trades)

    def test_snapshots_df_columns(self) -> None:
        sim = Simulation(seed=42)
        sim.run(500)
        df = sim.snapshots_df()
        for col in ["timestamp", "mid", "best_bid", "best_ask", "spread"]:
            assert col in df.columns

    def test_empty_trades_df_has_columns(self) -> None:
        sim = Simulation(seed=42)
        df = sim.trades_df()
        assert len(df) == 0
        assert "price" in df.columns
    
class TestStrategyIntegration:
    def _factory(self, engine, book):
        from lob.strategy import FixedSpreadMarketMaker
        return FixedSpreadMarketMaker(
            engine=engine, book=book,
            half_spread_ticks=2, quote_size=20, tick_size=0.01,
        )

    def test_runs_without_strategy(self) -> None:
        sim = Simulation(seed=42)
        sim.run(500)
        assert sim.strategy is None
        assert sim.strategy_snapshots == []

    def test_strategy_attached_via_factory(self) -> None:
        sim = Simulation(seed=42, strategy_factory=self._factory)
        assert sim.strategy is not None

    def test_strategy_receives_events(self) -> None:
        sim = Simulation(seed=42, strategy_factory=self._factory)
        sim.run(2_000)
        # The strategy should have been quoting and getting filled
        assert len(sim.strategy.fills) > 0

    def test_strategy_snapshots_captured(self) -> None:
        sim = Simulation(seed=42, strategy_factory=self._factory)
        sim.run(2_000)
        snaps = sim.strategy_snapshots_df()
        assert len(snaps) > 5
        for col in ["timestamp", "inventory", "cash", "pnl", "n_fills"]:
            assert col in snaps.columns

    def test_strategy_orders_are_protected_from_noise_cancels(self) -> None:
        sim = Simulation(seed=42, strategy_factory=self._factory)
        sim.run(2_000)
        # The strategy's own cancellations are voluntary (re-quoting). It
        # shouldn't be losing quotes constantly to the noise simulator.
        # We check that all cancelled orders are non-strategy IDs (< 1M).
        # This is indirect — we check that simulation cancels never failed
        # because the order was in the strategy's hands.
        stats = sim.stats()
        assert stats.cancels_succeeded == stats.cancels_attempted