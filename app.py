# app.py — LRS SMA/EMA 回測系統（含交易次數年度統計）
import os
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
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
st.markdown("<h1 style='margin-bottom:0.5em;'>📊 Leverage Rotation Strategy — SMA / EMA 回測系統</h1>", unsafe_allow_html=True)

# === 使用者輸入 ===
col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.text_input("輸入代號（例：00631L.TW, QQQ, SPXL, BTC-USD）", "00631L.TW")
with col2:
    start = st.date_input("開始日期", pd.to_datetime("2023-01-01"))
with col3:
    end = st.date_input("結束日期", pd.to_datetime("2025-01-01"))

col4, col5 = st.columns(2)
with col4:
    ma_type = st.selectbox("均線種類", ["SMA", "EMA"])
with col5:
    window = st.slider("均線天數", 50, 200, 200, 10)

# === 主回測流程 ===
if st.button("開始回測 🚀"):
    start_early = pd.to_datetime(start) - pd.Timedelta(days=365)
    with st.spinner("資料下載中…（自動多抓一年暖機資料）"):
        df_raw = yf.download(symbol, start=start_early, end=end)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

    if df_raw.empty or "Close" not in df_raw:
        st.error("⚠️ 無法下載資料，請檢查代號或時間區間。")
        st.stop()

    df = df_raw.copy()
    df["MA"] = (
        df["Close"].rolling(window=window).mean()
        if ma_type == "SMA"
        else df["Close"].ewm(span=window, adjust=False).mean()
    )
    df["Signal"] = np.where(df["Close"] > df["MA"], 1, 0)
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Position"] = df["Signal"].shift(1).fillna(0)
    df["Strategy_Return"] = df["Return"] * df["Position"]

    df["Equity_LRS"] = (1 + df["Strategy_Return"]).cumprod()
    df["Equity_BuyHold"] = (1 + df["Return"]).cumprod()
    df = df.loc[pd.to_datetime(start): pd.to_datetime(end)].copy()
    df["Equity_LRS"] /= df["Equity_LRS"].iloc[0]
    df["Equity_BuyHold"] /= df["Equity_BuyHold"].iloc[0]

    # === 買賣點 ===
    buy_points, sell_points = [], []
    prev_signal = None
    for i in range(len(df)):
        signal = int(df["Signal"].iloc[i])
        price = float(df["Close"].iloc[i])
        if prev_signal is None:
            prev_signal = signal
            continue
        if signal == 1 and prev_signal == 0:
            buy_points.append((df.index[i], price))
        elif signal == 0 and prev_signal == 1:
            sell_points.append((df.index[i], price))
        prev_signal = signal

    buy_count, sell_count = len(buy_points), len(sell_points)

    # === 年度交易統計 ===
    df["Year"] = df.index.year
    yearly_trade = pd.DataFrame({
        "年份": sorted(df["Year"].unique()),
        "買進次數": [len([b for b in buy_points if b[0].year == y]) for y in sorted(df["Year"].unique())],
        "賣出次數": [len([s for s in sell_points if s[0].year == y]) for y in sorted(df["Year"].unique())],
    })
    yearly_trade["總交易次數"] = yearly_trade["買進次數"] + yearly_trade["賣出次數"]

    # === 績效計算 ===
    final_return_lrs = df["Equity_LRS"].iloc[-1] - 1
    final_return_bh = df["Equity_BuyHold"].iloc[-1] - 1
    years_len = max((df.index[-1] - df.index[0]).days / 365, 1e-9)
    cagr_lrs = (1 + final_return_lrs) ** (1 / years_len) - 1
    cagr_bh = (1 + final_return_bh) ** (1 / years_len) - 1
    mdd_lrs = 1 - (df["Equity_LRS"] / df["Equity_LRS"].cummax()).min()
    mdd_bh = 1 - (df["Equity_BuyHold"] / df["Equity_BuyHold"].cummax()).min()

    def calc_metrics(series):
        daily = series.dropna()
        avg = daily.mean()
        std = daily.std()
        downside_std = daily[daily < 0].std()
        vol = std * np.sqrt(252)
        sharpe = (avg / std) * np.sqrt(252) if std > 0 else np.nan
        sortino = (avg / downside_std) * np.sqrt(252) if downside_std > 0 else np.nan
        return vol, sharpe, sortino

    vol_lrs, sharpe_lrs, sortino_lrs = calc_metrics(df["Strategy_Return"])
    vol_bh, sharpe_bh, sortino_bh = calc_metrics(df["Return"])

    # === 風控 ===
    loss_streak = (df["Strategy_Return"] < 0).astype(int)
    max_consecutive_loss = loss_streak.groupby(loss_streak.diff().ne(0).cumsum()).transform("size")[loss_streak == 1].max()
    flat_days = (df["Position"] == 0).astype(int)
    max_flat_days = flat_days.groupby(flat_days.diff().ne(0).cumsum()).transform("size")[flat_days == 1].max()

    # === 綜合報表 ===
    st.markdown("<h2 style='margin-top:1.5em;'>📊 綜合回測績效報表</h2>", unsafe_allow_html=True)
    summary_data = {
        "指標": [
            "總報酬", "年化報酬", "最大回撤", "年化波動率",
            "夏普值", "索提諾值", "最大連續虧損天數", "最長空倉天數",
            "買進次數", "賣出次數"
        ],
        "LRS": [
            f"{final_return_lrs:.2%}", f"{cagr_lrs:.2%}", f"{mdd_lrs:.2%}",
            f"{vol_lrs:.2%}", f"{sharpe_lrs:.2f}", f"{sortino_lrs:.2f}",
            f"{int(max_consecutive_loss)} 天", f"{int(max_flat_days)} 天",
            f"{buy_count}", f"{sell_count}"
        ],
        "Buy&Hold": [
            f"{final_return_bh:.2%}", f"{cagr_bh:.2%}", f"{mdd_bh:.2%}",
            f"{vol_bh:.2%}", f"{sharpe_bh:.2f}", f"{sortino_bh:.2f}",
            "—", "—", "—", "—"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

    # === 年度交易次數柱狀圖 ===
    st.markdown("<h3 style='margin-top:2em;'>📊 年度交易次數統計</h3>", unsafe_allow_html=True)
    fig_trade = go.Figure()
    fig_trade.add_trace(go.Bar(x=yearly_trade["年份"], y=yearly_trade["買進次數"], name="買進", marker_color="#27AE60"))
    fig_trade.add_trace(go.Bar(x=yearly_trade["年份"], y=yearly_trade["賣出次數"], name="賣出", marker_color="#E74C3C"))
    fig_trade.update_layout(barmode="group", template="plotly_white", height=400, xaxis_title="年份", yaxis_title="次數")
    st.plotly_chart(fig_trade, use_container_width=True)

    # === 年報酬折線、月熱力、年摘要 ===（省略顯示，與前版相同）
    # …（保持你上一版邏輯即可）
