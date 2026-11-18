# Crypto Futures Scanner

A signal-only Streamlit dashboard using public Binance USDⓈ-M Futures market data.

## Run locally
1. Install Python 3.10+
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

No Binance API key is required. This app does not place orders.

## Deploy free
Create a GitHub repository, upload `app.py` and `requirements.txt`, then deploy the repository with Streamlit Community Cloud.

## Important limitations
- This is an initial prototype, not a validated trading system.
- Top 50 means eligible USDT perpetual contracts ranked by 24h quote volume.
- It scans sequentially and may take time or encounter API rate limits.
- The leverage selector is informational only; it does not calculate exchange-specific liquidation price or set account leverage.
- Signal levels are illustrative ATR-based levels and do not include fees, funding, spread or slippage.
- Backtest and paper-trade before considering any real trading.
