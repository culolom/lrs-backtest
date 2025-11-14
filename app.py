# app.py — LRS 回測系統（台股 FinMind + 美股 yfinance）
import os
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt

from FinMind.data import DataLoader

import matplotlib.font_manager as fm
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 字型設定 ===
font_path = "./NotoSansTC-Bold.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams["font.family"] = "Noto Sans TC"
else:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Heiti TC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# === Streamlit 頁面設定 ===
st.set_page_config(page_title="LRS 回測系統", page_icon="📈", layout="wide")
st.markdown("<h1 style='margin-bottom:0.5em;'>📊 Leverage Rotation Strategy — 台股/美股 回測</h1>", unsafe_allow_html=True)

# === FinMind Token ===
FINMIND_TOKEN = st.secrets.get("FINMIND_TOKEN", "")

dl = DataLoader()
if FINMIND_TOKEN:
    dl.login_by_token(FINMIND_TOKEN)


# === 判斷台股/美股 ===
def is_tw_stock(symbol):
    return symbol.isdigit() or symbol.endswith(".TW")


def normalize_symbol(symbol):
    s = symbol.strip().upper()
    if s.isdigit():
        return s  # 台股編號 2330
    if s.endswith(".TW"):
        return s.replace(".TW", "")
    return s  # 美股


# === 台股 (FinMind) ===
@st.cache_data(show_spinner=False)
def load_tw_stock(symbol, start, end):
    """回傳台股 OHLCV（已還原權息、拆股）"""
    df = dl.taiwan_stock_daily(
        stock_id=symbol,
        start_date=str(start),
        end_date=str(end),
    )
    if df.empty:
        st.error(f"⚠️ FinMind 無法取得 {symbol} 的資料")
        return pd.DataFrame()

    # 調整欄位
    df["Date"] = pd.to_datetime(df["date"])
    df = df.sort_values("Date")
    df = df.rename(columns={
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "Trading_Volume": "Volume",
    })

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].set_index("Date")
    return df


# === 美股 (yfinance) ===
@st.cache_data(show_spinner=False)
def load_us_stock(symbol, start, end):
    yf_symbol = symbol if "." not in symbol else symbol.replace(".TW", "")
    df = yf.download(yf_symbol, start=start, end=end, auto_adjust=True)
    if df.empty:
        st.error(f"⚠️ 無法取得美股 {symbol} 的資料")
    return df


# === 統一資料來源 ===
def load_price(symbol, start, end):
    symbol = normalize_symbol(symbol)

    if is_tw_stock(symbol):
        return load_tw_stock(symbol, start, end)
    else:
        return load_us_stock(symbol, start, end)


# === 使用者輸入 ===
col1, col2, col3 = st.columns(3)
with col1:
    raw_symbol = st.text_input("輸入商品代號 (例：0050、2330、QQQ、SPY)", "0050")

symbol = normalize_symbol(raw_symbol)

today = dt.date.today()
default_start = dt.date(2013, 1, 1)

with col2:
    start = st.date_input("開始日期", value=default_start)
with col3:
    end = st.date_input("結束日期", value=today)

col4, col5, col6 = st.columns(3)
with col4:
    ma_type = st.selectbox("均線種類", ["SMA", "EMA"])
with col5:
    window = st.slider("均線天數", 10, 200, 200, 10)
with col6:
    initial_capital = st.number_input("投入本金（元）", 1000, 5_000_000, 100000, step=1000)


# === 主程式 ===
if st.button("開始回測 🚀"):

    df = load_price(symbol, start, end)

    if df.empty:
        st.stop()

    df["MA"] = (
        df["Close"].rolling(window=window).mean()
        if ma_type == "SMA"
        else df["Close"].ewm(span=window, adjust=False).mean()
    )

    # === 訊號 ===
    df["Signal"] = 0
    df.iloc[0, df.columns.get_loc("Signal")] = 1
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["MA"].iloc[i] and df["Close"].iloc[i - 1] <= df["MA"].iloc[i - 1]:
            df.iloc[i, df.columns.get_loc("Signal")] = 1
        elif df["Close"].iloc[i] < df["MA"].iloc[i] and df["Close"].iloc[i - 1] >= df["MA"].iloc[i - 1]:
            df.iloc[i, df.columns.get_loc("Signal")] = -1

    # === 持倉 ===
    position = []
    curr = 1
    for sig in df["Signal"]:
        if sig == 1:
            curr = 1
        elif sig == -1:
            curr = 0
        position.append(curr)
    df["Position"] = position

    # === 報酬 ===
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Return"] * df["Position"]

    # === 資金曲線 ===
    df["Equity_LRS"] = (1 + df["Strategy_Return"]).cumprod()
    df["Equity_BH"] = (1 + df["Return"]).cumprod()

    # === 買賣點 ===
    buy_points = [(i, df["Close"].iloc[i]) for i in range(1, len(df)) if df["Signal"].iloc[i] == 1]
    sell_points = [(i, df["Close"].iloc[i]) for i in range(1, len(df)) if df["Signal"].iloc[i] == -1]

    # === 圖表 ===
    st.markdown("## 📈 策略績效視覺化")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("收盤價與均線（含買賣點）", "資金曲線：LRS vs Buy&Hold"))

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="收盤價", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA"], name=f"{ma_type}{window}", line=dict(color="orange")), row=1, col=1)

    if buy_points:
        x, y = zip(*[(df.index[i], p) for i, p in buy_points])
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="買進",
                                 marker=dict(color="green", symbol="triangle-up", size=8)), row=1, col=1)

    if sell_points:
        x, y = zip(*[(df.index[i], p) for i, p in sell_points])
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="賣出",
                                 marker=dict(color="red", symbol="x", size=8)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_LRS"], name="LRS 策略", line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_BH"], name="Buy & Hold",
                             line=dict(color="gray", dash="dot")), row=2, col=1)

    fig.update_layout(height=800, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ 回測完成！台股 FinMind + 美股 yfinance 已啟用")

