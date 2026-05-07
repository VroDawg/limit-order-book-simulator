"""Interactive Streamlit app for the LOB simulator.

Run locally:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from lob.metrics import compute_metrics
from lob.order_flow import OrderFlowParams
from lob.plotting import plot_backtest
from lob.simulation import Simulation
from lob.strategy import (
    AvellanedaStoikovMarketMaker,
    FixedSpreadMarketMaker,
    InventoryAwareMarketMaker,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LOB Simulator",
    page_icon="📊",
    layout="wide",
)

st.title("Limit Order Book Simulator")
st.markdown(
    "Interactive backtester for market-making strategies. "
    "Pick a strategy, tune its parameters and the market regime, "
    "then watch how the strategy behaves over a simulated trading session."
)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Simulation parameters")

    n_events = st.slider("Number of events", 2_000, 50_000, 20_000, step=1_000)
    seed = st.number_input("Random seed", value=42, min_value=0, step=1)

    st.subheader("Market regime")
    regime = st.selectbox(
        "Regime",
        ["Balanced", "Bearish", "Bullish", "Custom"],
        help="Bearish = more market sells than buys. Causes downward drift.",
    )
    if regime == "Balanced":
        lambda_market_buy, lambda_market_sell = 2e-8, 2e-8
    elif regime == "Bearish":
        lambda_market_buy, lambda_market_sell = 1e-8, 5e-8
    elif regime == "Bullish":
        lambda_market_buy, lambda_market_sell = 5e-8, 1e-8
    else:
        lambda_market_buy = st.slider(
            "Market buy rate (×1e-8)", 0.5, 10.0, 2.0, step=0.5
        ) * 1e-8
        lambda_market_sell = st.slider(
            "Market sell rate (×1e-8)", 0.5, 10.0, 2.0, step=0.5
        ) * 1e-8

    st.subheader("Strategy")
    strategy_name = st.selectbox(
        "Strategy",
        ["FixedSpread", "InventoryAware", "AvellanedaStoikov"],
    )
    quote_size = st.slider("Quote size (shares)", 5, 100, 20, step=5)

    if strategy_name == "FixedSpread":
        half_spread_ticks = st.slider("Half-spread (ticks)", 1, 10, 2)
    elif strategy_name == "InventoryAware":
        half_spread_ticks = st.slider("Half-spread (ticks)", 1, 10, 2)
        skew_per_share = st.number_input(
            "Skew per share ($)", value=0.0001, format="%.5f", step=0.00001,
            help="Quote shift per unit of inventory. Higher = more aggressive flattening.",
        )
    else:  # Avellaneda-Stoikov
        sigma = st.slider("σ (volatility per √s)", 0.01, 0.50, 0.10, step=0.01)
        gamma = st.slider("γ (risk aversion)", 0.001, 0.500, 0.050, step=0.005)
        kappa = st.slider("κ (arrival intensity)", 1.0, 200.0, 50.0, step=1.0)
        horizon_seconds = st.slider("Horizon (s)", 0.1, 10.0, 1.0, step=0.1)

    run_button = st.button(
        "▶ Run simulation", type="primary", use_container_width=True
    )


# ---------------------------------------------------------------------------
# Strategy factory (closure over sidebar values)
# ---------------------------------------------------------------------------
def build_strategy(engine, book):
    if strategy_name == "FixedSpread":
        return FixedSpreadMarketMaker(
            engine, book,
            half_spread_ticks=half_spread_ticks,
            quote_size=quote_size, tick_size=0.01,
        )
    if strategy_name == "InventoryAware":
        return InventoryAwareMarketMaker(
            engine, book,
            half_spread_ticks=half_spread_ticks,
            quote_size=quote_size, tick_size=0.01,
            skew_per_share=skew_per_share,
        )
    return AvellanedaStoikovMarketMaker(
        engine, book,
        sigma=sigma, gamma=gamma, kappa=kappa,
        horizon_seconds=horizon_seconds,
        quote_size=quote_size, tick_size=0.01,
    )


# ---------------------------------------------------------------------------
# Run on click (or first load)
# ---------------------------------------------------------------------------
if run_button or "sim_result" not in st.session_state:
    with st.spinner("Running simulation..."):
        sim = Simulation(
            params=OrderFlowParams(
                reference_price=100.0,
                lambda_market_buy=lambda_market_buy,
                lambda_market_sell=lambda_market_sell,
            ),
            strategy_factory=build_strategy,
            snapshot_interval_ns=50_000_000,
            seed=int(seed),
        )
        sim.run(n_events=n_events)
        st.session_state.sim_result = (sim, compute_metrics(sim))
        st.session_state.last_strategy_name = strategy_name
        st.session_state.last_regime = regime

sim, metrics = st.session_state.sim_result

# ---------------------------------------------------------------------------
# Main area — metrics + dashboard
# ---------------------------------------------------------------------------
st.header(
    f"Results — {st.session_state.last_strategy_name} "
    f"({st.session_state.last_regime})"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Final P&L",       f"${metrics.final_pnl:,.2f}")
c2.metric("Sharpe (per snap)", f"{metrics.sharpe_per_snapshot:.4f}")
c3.metric("Max drawdown",    f"${metrics.max_drawdown:,.2f}")
c4.metric("Total fills",     f"{metrics.n_fills:,}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Max abs inventory", f"{metrics.inventory_max:,}")
c2.metric("Inventory std",     f"{metrics.inventory_std:.2f}")
c3.metric("Realized spread",   f"${metrics.avg_realized_spread:.5f}")
c4.metric("Adverse selection", f"${metrics.avg_adverse_selection:.5f}")

st.divider()

st.subheader("Backtest dashboard")
fig = plot_backtest(sim, show=False)
st.pyplot(fig)
plt.close(fig)

with st.expander("More about this simulator"):
    st.markdown(
    """ This application runs a small market simulation. Indeed, orders arrive at random
        times on both sides of an order book, and a matching engine, pairs
        them up using price-time priority, the same matching logic real
        exchanges use. A market-making strategy participates alongside the
        random flow, posting bids and asks to capture the spread between
        them.

        The questions this application seeks to explore are the following: "does the strategy actually
        make money, and also, how do its choices about quote width and inventory
        management change the outcome?"

        "FixedSpread" is the simplest possible market maker. It quotes a
        fixed distance from the mid-price and never thinks about its
        inventory. In a falling market it will happily accumulate a
        thousand-share long position before noticing anything is wrong.

        "InventoryAware" adds the obvious next idea. When inventory
        grows, both quotes shift in the direction that makes the
        offsetting trade more likely. If the strategy is long, both bid
        and ask drop, making the ask more attractive to buyers. Inventory
        drifts back toward zero on its own.

        "AvellanedaStoikov" is the 2008 academic formulation. Given the following:
        volatility (σ), risk aversion (γ), order arrival intensity (κ),
        and a time horizon, it analytically derives both the optimal
        quote skew and the optimal half-spread. It's the standard academic
        reference for market-making strategy.

        Furthermore, the regime selector matters because no strategy is universally
        good. In a balanced market, the simple one often wins on raw P&L
        because it captures spread without paying for risk insurance it
        doesn't need. In a trending market, that same simplicity becomes
        a liability. The strategy accumulates toxic inventory and gets
        run over by the drift.
        """
    )
    
    