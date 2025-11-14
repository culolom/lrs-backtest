# app.py — LRS 回測系統（台股用 HWTR API，美股用 yfinance）
# Ver.2025.02 — 完整修正版，可直接部署

import os
import math
import datetime as dt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
import matplotlib.font_manager as fm

# -----------------------------------------------------------------------------
# 字型設定
# -----------------------------------------------------------------------------
font_path = "./NotoSansTC-Bold.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams["font.family"] = "Noto Sans TC"
else:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Heiti TC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# -----------------------------------------------------------------------------
# Streamlit UI 設定
# -----------------------------------------------------------------------------
st.set_page_config(page_title="LRS 回測系統", page_icon="📈", layout="wide")
st.markdown("<h1 style='margin-bottom:0.5em;'>📊 Leverage Rotation Strategy — SMA/EMA 回測系統</h1>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 工具：判斷台股/美股
# -----------------------------------------------------------------------------
def is_taiwan_stock(symbol: str) -> bool:
    """
    台股代號：全部是數字 & <=4 字 (0050 / 2330 / 00878)
    """
    s = symbol.strip().upper()
    return s.isdigit() and len(s) <= 4


def normalize_yf(symbol: str) -> str:
    """
    美股用原代號 (QQQ)
    台股自動加上 .TW（給 yfinance 用，只在美股模式時）
    """
    s = symbol.strip().upper()
    if is_taiwan_stock(s):
        return s + ".TW"
    return s

# -----------------------------------------------------------------------------
# HWTR API：讀取台股資料
# -----------------------------------------------------------------------------
def fetch_hwtr_history(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    使用 HWTR API 下載 OHLCV
    回傳欄位：Open, High, Low, Close, Volume
    """
    url = "https://api.hwtrader.com/stock/history"
    params = {
        "symbol": symbol,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        js = r.json()
    except Exception:
        return pd.DataFrame()

    if "data" not in js or len(js["data"]) == 0:
        return pd.DataFrame()

    rows = []
    for row in js["data"]:
        # row = ['2023-01-02', open, high, low, close, volume]
        try:
            d = pd.to_datetime(row[0])
            o, h, l, c, v = row[1], row[2], row[3], row[4], row[5]
            rows.append(
                {"Date": d, "Open": o, "High": h, "Low": l, "Close": c, "Adj Close": c, "Volume": v}
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df

# -----------------------------------------------------------------------------
# yfinance：下載海外商品
# -----------------------------------------------------------------------------
def fetch_yf_history(yf_symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(yf_symbol, start=start, end=end, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return df
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df

# -----------------------------------------------------------------------------
# 垂直斷崖修正（拆股）
# -----------------------------------------------------------------------------
def adjust_splits(df: pd.DataFrame, price_col="Close", threshold=0.28) -> pd.DataFrame:
    """
    遇到 pct_change 超過 threshold (例如 -70%) 視為拆股
    自動把「之前所有歷史價格」乘上比例，使歷史曲線連續
    """

    if df.empty:
        return df

    df = df.copy()
    df["Price_raw"] = df[price_col].astype(float)
    df["Price_adj"] = df["Price_raw"].copy()

    pct = df["Price_raw"].pct_change()

    # 找出極異常跳水
    events = pct[pct <= -threshold].dropna()

    for date, r in events.items():
        ratio = 1 + r  # 例如：-0.75 → 0.25 → 4:1 split
        if ratio <= 0:
            continue
        # 對 date 以前全部調整
        df.loc[df.index < date, "Price_adj"] *= ratio

    return df

# -----------------------------------------------------------------------------
# 統一載入價格：台股 → HWTR，美股 → yfinance
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_price(symbol: str, yf_symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:

    if is_taiwan_stock(symbol):
        df = fetch_hwtr_history(symbol, start, end)
        if df.empty:
            raise ValueError(f"HWTR API 無法取得 {symbol} 的資料")

        df = adjust_splits(df, "Close")
        df["Price"] = df["Price_adj"]
        return df

    else:
        df = fetch_yf_history(yf_symbol, start, end)
        if df.empty:
            raise ValueError(f"yfinance 無法取得 {yf_symbol}")

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        df = adjust_splits(df, price_col)
        df["Price"] = df["Price_adj"]
        return df

# -----------------------------------------------------------------------------
# UI：輸入
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    raw_symbol = st.text_input("輸入代號（例：0050 / 2330 / 00878 / QQQ）", "0050")

yf_symbol = normalize_yf(raw_symbol)

# 日期區間：全部放寬（HWTR 只抓指定區間）
with col2:
    start = st.date_input("開始日期", value=dt.date(2013, 1, 1))
with col3:
    end = st.date_input("結束日期", value=dt.date.today())

col4, col5, col6 = st.columns(3)
with col4:
    ma_type = st.selectbox("均線種類", ["SMA", "EMA"])
with col5:
    window = st.slider("均線天數", 10, 200, 200, 10)
with col6:
    initial_capital = st.number_input("投入本金（元）", 10000, 1_000_000, 10000, step=10000)

# -----------------------------------------------------------------------------
# 主計算
# -----------------------------------------------------------------------------
if st.button("開始回測 🚀"):

    with st.spinner("資料下載中…"):
        start_early = pd.to_datetime(start) - pd.Timedelta(days=365)
        df_all = load_price(raw_symbol, yf_symbol, start_early.date(), end)

    if df_all.empty:
        st.error("⚠️ 資料下載失敗")
        st.stop()

    df = df_all.copy()
    df = df[(df.index >= pd.to_datetime(start_early)) & (df.index <= pd.to_datetime(end))]

    # 均線
    if ma_type == "SMA":
        df["MA"] = df["Price"].rolling(window=window).mean()
    else:
        df["MA"] = df["Price"].ewm(span=window, adjust=False).mean()

    df = df.dropna(subset=["MA"]).copy()

    # 訊號：第一天強制買進
    df["Signal"] = 0
    df.iloc[0, df.columns.get_loc("Signal")] = 1

    for i in range(1, len(df)):
        p, m = df["Price"].iloc[i], df["MA"].iloc[i]
        p_last, m_last = df["Price"].iloc[i - 1], df["MA"].iloc[i - 1]

        if p > m and p_last <= m_last:
            df.iloc[i, df.columns.get_loc("Signal")] = 1
        elif p < m and p_last >= m_last:
            df.iloc[i, df.columns.get_loc("Signal")] = -1

    # 持倉
    current = 1
    position = []
    for sig in df["Signal"]:
        if sig == 1:
            current = 1
        elif sig == -1:
            current = 0
        position.append(current)
    df["Position"] = position

    # 日報酬
    df["Return"] = df["Price"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Return"] * df["Position"]

    # 資金曲線（LRS）
    df["Equity_LRS"] = (1 + df["Strategy_Return"]).cumprod()

    # Buy & Hold
    df["Equity_BuyHold"] = (1 + df["Return"]).cumprod()

    # 改為使用者選的區間（重新歸一化）
    df = df.loc[pd.to_datetime(start): pd.to_datetime(end)].copy()
    df["Equity_LRS"] /= df["Equity_LRS"].iloc[0]
    df["Equity_BuyHold"] /= df["Equity_BuyHold"].iloc[0]

    df["LRS_Capital"] = df["Equity_LRS"] * initial_capital
    df["BH_Capital"] = df["Equity_BuyHold"] * initial_capital

    # 買賣點
    buy_points = [(df.index[i], df["Price"].iloc[i]) for i in range(1, len(df)) if df["Signal"].iloc[i] == 1]
    sell_points = [(df.index[i], df["Price"].iloc[i]) for i in range(1, len(df)) if df["Signal"].iloc[i] == -1]

    # 指標
    final_return_lrs = df["Equity_LRS"].iloc[-1] - 1
    final_return_bh = df["Equity_BuyHold"].iloc[-1] - 1
    years_len = (df.index[-1] - df.index[0]).days / 365

    cagr_lrs = (1 + final_return_lrs) ** (1 / years_len) - 1
    cagr_bh = (1 + final_return_bh) ** (1 / years_len) - 1

    mdd_lrs = 1 - (df["Equity_LRS"] / df["Equity_LRS"].cummax()).min()
    mdd_bh = 1 - (df["Equity_BuyHold"] / df["Equity_BuyHold"].cummax()).min()

    def calc_stats(series):
        avg = series.mean()
        std = series.std()
        downside = series[series < 0].std()
        vol = std * np.sqrt(252)
        sharpe = (avg / std) * np.sqrt(252) if std > 0 else np.nan
        sortino = (avg / downside) * np.sqrt(252) if downside > 0 else np.nan
        return vol, sharpe, sortino

    vol_lrs, sharpe_lrs, sortino_lrs = calc_stats(df["Strategy_Return"])
    vol_bh, sharpe_bh, sortino_bh = calc_stats(df["Return"])

    # -----------------------------------------------------------------------------
    # 圖表
    # -----------------------------------------------------------------------------
    st.markdown("<h2 style='margin-top:1em;'>📈 策略績效視覺化</h2>", unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("價格與均線（含買賣點）", "資金曲線：LRS vs Buy&Hold"))

    fig.add_trace(go.Scatter(x=df.index, y=df["Price"], name="價格", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA"], name=f"{ma_type}{window}", line=dict(color="orange")), row=1, col=1)

    if buy_points:
        bx, by = zip(*buy_points)
        fig.add_trace(go.Scatter(x=bx, y=by, mode="markers", name="買進",
                                 marker=dict(color="green", symbol="triangle-up", size=8)), row=1, col=1)

    if sell_points:
        sx, sy = zip(*sell_points)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers", name="賣出",
                                 marker=dict(color="red", symbol="x", size=8)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_LRS"], name="LRS 策略", line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_BuyHold"], name="Buy & Hold",
                             line=dict(color="gray", dash="dot")), row=2, col=1)

    fig.update_layout(height=800, showlegend=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------------
    # 報表
    # -----------------------------------------------------------------------------
    st.markdown("""
    <style>
    .custom-table {
        width:100%; border-collapse:collapse; margin-top:1.2em; font-family:"Noto Sans TC";
    }
    .custom-table th {
        background:#f5f6fa; padding:12px; font-weight:700; border-bottom:2px solid #ddd;
    }
    .custom-table td {
        text-align:center; padding:10px; border-bottom:1px solid #eee; font-size:15px;
    }
    .custom-table tr:nth-child(even) td { background-color:#fafbfc; }
    .custom-table tr:hover td { background-color:#f1f9ff; }
    .section-title td {
        background:#eef4ff; color:#1a237e; font-weight:700; font-size:16px;
        text-align:left; padding:10px 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    html_table = f"""
    <table class='custom-table'>
    <thead><tr><th>指標名稱</th><th>LRS 策略</th><th>Buy & Hold</th></tr></thead>
    <tbody>
    <tr><td>最終資產</td><td>{df['LRS_Capital'].iloc[-1]:,.0f} 元</td>
        <td>{df['BH_Capital'].iloc[-1]:,.0f} 元</td></tr>
    <tr><td>總報酬</td><td>{final_return_lrs:.2%}</td><td>{final_return_bh:.2%}</td></tr>
    <tr><td>年化報酬</td><td>{cagr_lrs:.2%}</td><td>{cagr_bh:.2%}</td></tr>
    <tr><td>最大回撤</td><td>{mdd_lrs:.2%}</td><td>{mdd_bh:.2%}</td></tr>
    <tr><td>年化波動率</td><td>{vol_lrs:.2%}</td><td>{vol_bh:.2%}</td></tr>
    <tr><td>夏普值</td><td>{sharpe_lrs:.2f}</td><td>{sharpe_bh:.2f}</td></tr>
    <tr><td>索提諾值</td><td>{sortino_lrs:.2f}</td><td>{sortino_bh:.2f}</td></tr>

    <tr class='section-title'><td colspan='3'>💹 交易統計</td></tr>
    <tr><td>買進次數</td><td>{len(buy_points)}</td><td>-</td></tr>
    <tr><td>賣出次數</td><td>{len(sell_points)}</td><td>-</td></tr>

    </tbody></table>
    """

    st.markdown(html_table, unsafe_allow_html=True)
    st.success("✅ 回測完成！台股使用 HWTR API，自動拆股修正避免斷崖")
