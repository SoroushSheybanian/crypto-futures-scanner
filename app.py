import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh

BASE = "https://www.okx.com"
TIMEOUT = 12

st.set_page_config(page_title="Crypto Futures Scanner", page_icon="📈", layout="wide")



@st.cache_data(ttl=60)
def get_universe(limit=50):
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 CryptoScanner/1.0"
    })

    # Get active USDT perpetual swap instruments
    r = session.get(
        f"{BASE}/api/v5/public/instruments",
        params={"instType": "SWAP"},
        timeout=TIMEOUT
    )
    r.raise_for_status()
    info = r.json()

    if info.get("code") != "0":
        raise RuntimeError(f"OKX instruments error: {info}")

    eligible = {
        s["instId"]
        for s in info.get("data", [])
        if s.get("state") == "live"
        and s.get("settleCcy") == "USDT"
        and s.get("ctType") == "linear"
        and s.get("instId", "").endswith("-USDT-SWAP")
    }

    # Get 24-hour ticker data
    r = session.get(
        f"{BASE}/api/v5/market/tickers",
        params={"instType": "SWAP"},
        timeout=TIMEOUT
    )
    r.raise_for_status()
    tickers = r.json()

    if tickers.get("code") != "0":
        raise RuntimeError(f"OKX ticker error: {tickers}")

    rows = [
        x for x in tickers.get("data", [])
        if x.get("instId") in eligible
    ]

    rows.sort(
        key=lambda x: float(
            x.get("volCcy24h") or x.get("vol24h") or 0
        ),
        reverse=True
    )

    result = [x["instId"] for x in rows[:limit]]

    if not result:
        raise RuntimeError(
            "No active USDT perpetual markets found on OKX."
        )

    return result



@st.cache_data(ttl=20)
def get_klines(symbol, interval, limit=240):
    r = requests.get(
        f"{BASE}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def indicators(df):
    d = df.copy()
    d["ema20"] = d.close.ewm(span=20, adjust=False).mean()
    d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
    d["ema200"] = d.close.ewm(span=200, adjust=False).mean()
    delta = d.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))
    fast = d.close.ewm(span=12, adjust=False).mean()
    slow = d.close.ewm(span=26, adjust=False).mean()
    d["macd"] = fast - slow
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()
    d["vol_avg"] = d.volume.rolling(20).mean()
    return d

def timeframe_bias(symbol, interval):
    d = indicators(get_klines(symbol, interval))
    # Ignore the currently forming candle.
    x = d.iloc[-2]
    bullish = x.close > x.ema20 > x.ema50 and x.macd > x.macd_signal and 50 <= x.rsi <= 72
    bearish = x.close < x.ema20 < x.ema50 and x.macd < x.macd_signal and 28 <= x.rsi <= 50
    if bullish:
        return "Bullish", x
    if bearish:
        return "Bearish", x
    return "Neutral", x

def analyze(symbol):
    try:
        b5, x5 = timeframe_bias(symbol, "5m")
        b15, x15 = timeframe_bias(symbol, "15m")
        direction = "WAIT"
        if b5 == "Bullish" and b15 == "Bullish":
            direction = "LONG"
        elif b5 == "Bearish" and b15 == "Bearish":
            direction = "SHORT"

        price = float(x5.close)
        # ATR-based stop and targets using closed M5 candles.
        d = indicators(get_klines(symbol, "5m"))
        d["tr"] = np.maximum(d.high-d.low,
                    np.maximum((d.high-d.close.shift()).abs(), (d.low-d.close.shift()).abs()))
        atr = float(d.tr.rolling(14).mean().iloc[-2])
        if not np.isfinite(atr) or atr <= 0:
            atr = price * 0.005

        if direction == "LONG":
            sl = price - 1.5 * atr
            risk = price - sl
            tps = [price + risk, price + 2*risk, price + 3*risk]
        elif direction == "SHORT":
            sl = price + 1.5 * atr
            risk = sl - price
            tps = [price - risk, price - 2*risk, price - 3*risk]
        else:
            sl, tps = np.nan, [np.nan]*3

        return {
            "Symbol": symbol, "Price": price, "M5": b5, "M15": b15,
            "Signal": direction, "Entry": price, "Stop Loss": sl,
            "TP1": tps[0], "TP2": tps[1], "TP3": tps[2],
            "Updated": pd.Timestamp.utcnow().strftime("%H:%M:%S UTC"), "Error": ""
        }
    except Exception as e:
        return {"Symbol": symbol, "Price": np.nan, "M5": "-", "M15": "-",
                "Signal": "ERROR", "Entry": np.nan, "Stop Loss": np.nan,
                "TP1": np.nan, "TP2": np.nan, "TP3": np.nan,
                "Updated": "-", "Error": str(e)[:100]}

st.title("📈 Crypto Futures Scanner")
st.caption("Binance USDⓈ-M perpetuals · Public market data · Signal-only prototype")

with st.sidebar:
    st.header("Scanner settings")
    leverage = st.select_slider("Leverage scenario", options=[3, 5, 10, 20, 30, 50, 75, 100], value=3)
    refresh = st.selectbox("Refresh interval", [30, 60, 120, 300], index=1, format_func=lambda x: f"{x} seconds")
    st.caption("Leverage is a display scenario only. No orders are placed.")
    st.caption("This prototype uses a simple EMA/RSI/MACD + M5/M15 alignment strategy.")

st_autorefresh(interval=refresh * 1000, key="refresh")

try:
    symbols = get_universe(50)
    st.caption(f"Universe: top 50 USDT perpetual contracts by 24h quote volume · Refresh: {refresh}s")
    progress = st.progress(0, text="Scanning markets...")
    rows = []
    for i, sym in enumerate(symbols):
        rows.append(analyze(sym))
        progress.progress((i+1)/len(symbols), text=f"Scanning {sym} ({i+1}/{len(symbols)})")
        time.sleep(0.03)
    progress.empty()
    df = pd.DataFrame(rows)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Markets monitored", len(df))
    c2.metric("Long signals", int((df.Signal == "LONG").sum()))
    c3.metric("Short signals", int((df.Signal == "SHORT").sum()))
    c4.metric("Selected leverage", f"{leverage}x")

    st.subheader("Market overview")
    view = df[["Symbol","Price","M5","M15","Signal","Entry","Stop Loss","TP1","TP2","TP3","Updated"]].copy()
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="%.6g"),
            "Entry": st.column_config.NumberColumn(format="%.6g"),
            "Stop Loss": st.column_config.NumberColumn(format="%.6g"),
            "TP1": st.column_config.NumberColumn(format="%.6g"),
            "TP2": st.column_config.NumberColumn(format="%.6g"),
            "TP3": st.column_config.NumberColumn(format="%.6g"),
        },
    )

    st.subheader("Selected market details")
    selected = st.selectbox("Choose a symbol", symbols)
    detail = df[df.Symbol == selected].iloc[0]
    a,b,c = st.columns(3)
    a.metric("Signal", detail.Signal)
    b.metric("M5 bias", detail.M5)
    c.metric("M15 bias", detail.M15)
    if detail.Signal in ("LONG", "SHORT"):
        st.write({
            "Entry": detail.Entry, "Stop Loss": detail["Stop Loss"],
            "TP1": detail.TP1, "TP2": detail.TP2, "TP3": detail.TP3,
            "Leverage scenario": f"{leverage}x",
        })
        st.info("Illustrative levels only. Validate spread, funding, fees, slippage and liquidation risk before any live trade.")
    elif detail.Signal == "ERROR":
        st.warning(f"Data issue for this symbol: {detail.Error}")
    else:
        st.info("No aligned M5/M15 signal at the moment.")

    st.subheader("Signal logic")
    st.markdown("""
    - **LONG:** M5 and M15 both bullish; close above EMA20 above EMA50; MACD above signal; RSI in the confirmation band.
    - **SHORT:** M5 and M15 both bearish; close below EMA20 below EMA50; MACD below signal; RSI in the confirmation band.
    - **WAIT:** timeframes do not align.
    - Stops and targets are illustrative ATR-based levels (1.5 ATR stop; 1R / 2R / 3R targets).
    """)
except Exception as e:
    st.error(f"Could not load Binance Futures market data: {e}")
    st.info("Check your internet connection or whether Binance Futures API is reachable from your network.")

st.divider()
st.caption("Educational prototype only — not financial advice. No API trading keys are used and no orders are executed. The 3x–100x selector does not change Binance account leverage.")
