# Limit Order Book Simulator

A Python implementation of a limit order book matching engine and market-making strategy backtester.

## Features (planned)
- Price-time priority matching engine
- Limit, market, and cancel order support
- Event-driven simulator with synthetic and replayed order flow
- Market-making strategies including inventory-aware Avellaneda-Stoikov
- Performance analytics: P&L, fill rate, inventory variance, adverse selection, Sharpe

## Setup
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Tests
```bash
pytest
```