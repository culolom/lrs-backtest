# app.py — LRS (SMA/EMA + Plotly + 買賣次數 + 預設一年暖機)

import os
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.font_manager as fm
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 中文字型設定（自動偵測 + 雲端相容） ===
font_path = "./NotoSansTC-Bold.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams["font.family"] = "Noto Sans TC"
else:
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK TC",
        "Microsoft JhengHei",
        "PingFang TC",
        "Heiti TC",
    ]
matplotlib.rcParams["axes.unicode_minus"] = False

# === Streamlit 基本設定 ===
st.set_page_config(page_title="LRS 移動平均回測系統", page_icon="📈", layout="wide")
st.title("📊 Leverage Rotation Strategy — SMA / EMA 回測系統")

# === 使用者輸入區 ===
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
    window = st.slider("均線天數", 50, 300, 200, 10)

# === 按下按鈕後回測 ===
if st.button("開始回測 🚀"):
    # 自動提前一年抓資料（365 天暖機）
    start_early = pd.to_datetime(start) - pd.Timedelta(days=365)

    with st.spinner("資料下載中…（自動多抓一年暖機資料）"):
        df_raw = yf.download(symbol, start=start_early, end=end)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

    if df_raw.empty or "Close" not in df_raw:
        st.error("⚠️ 無法下載資料，請檢查代號或時間區間。")
        st.stop()

    # === 計算移動平均線 ===
    df = df_raw.copy()
    if ma_type == "SMA":
        df["MA"] = df["Close"].rolling(window=window).mean()
    else:
        df["MA"] = df["Close"].ewm(span=window, adjust=False).mean()

    # === 建立訊號與績效 ===
    df["Signal"] = np.where(df["Close"] > df["MA"], 1, 0)
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Return"] * df["Signal"]
    df["Equity_LRS"] = (1 + df["Strategy_Return"]).cumprod()
    df["Equity_BuyHold"] = (1 + df["Return"]).cumprod()

    # === 裁掉暖機期，只顯示設定的開始日期後的部分 ===
    df = df.loc[pd.to_datetime(start): pd.to_datetime(end)]

    # === 計算績效指標 ===
    final_return_lrs = df["Equity_LRS"].iloc[-1] - 1
    final_return_bh = df["Equity_BuyHold"].iloc[-1] - 1
    years = max((df.index[-1] - df.index[0]).days / 365, 1e-9)
    cagr_lrs = (1 + final_return_lrs) ** (1 / years) - 1
    cagr_bh = (1 + final_return_bh) ** (1 / years) - 1
    mdd_lrs = 1 - (df["Equity_LRS"] / df["Equity_LRS"].cummax()).min()
    mdd_bh = 1 - (df["Equity_BuyHold"] / df["Equity_BuyHold"].cummax()).min()

    # === 建立買賣點 ===
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

    buy_count = len(buy_points)
    sell_count = len(sell_points)

    # === Plotly 圖表 ===
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=(f"{symbol} {ma_type}{window} 買賣訊號", "策略績效對比"),
        vertical_spacing=0.1,
    )

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines",
        name="收盤價", line=dict(color="#2E86AB", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA"], mode="lines",
        name=f"{ma_type}{window}", line=dict(color="#F39C12", width=2)), row=1, col=1)

    if buy_points:
        bx, by = zip(*buy_points)
        fig.add_trace(go.Scatter(x=bx, y=by, mode="markers", name="買進",
                                 marker=dict(color="#27AE60", size=9, symbol="triangle-up")), row=1, col=1)
    if sell_points:
        sx, sy = zip(*sell_points)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers", name="賣出",
                                 marker=dict(color="#E74C3C", size=9, symbol="x")), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Equity_LRS"], mode="lines",
        name=f"LRS 策略 ({ma_type}{window})",
        line=dict(color="#16A085", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Equity_BuyHold"], mode="lines",
        name="Buy & Hold",
        line=dict(color="#7F8C8D", width=2, dash="dot")), row=2, col=1)

    fig.update_layout(
        height=700,
        template="plotly_white",
        title=dict(text=f"📈 {symbol} — {ma_type}{window} 回測（自動暖機一年）", x=0.5, font=dict(size=20)),
        legend=dict(orientation="h", y=-0.25),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # === 總體績效 ===
    st.subheader("📊 回測績效摘要")
    col1, col2, col3 = st.columns(3)
    col1.metric("LRS 總報酬", f"{final_return_lrs:.2%}")
    col2.metric("LRS 年化報酬", f"{cagr_lrs:.2%}")
    col3.metric("LRS 最大回撤", f"{mdd_lrs:.2%}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Buy&Hold 總報酬", f"{final_return_bh:.2%}")
    col5.metric("Buy&Hold 年化報酬", f"{cagr_bh:.2%}")
    col6.metric("Buy&Hold 最大回撤", f"{mdd_bh:.2%}")

    # === 買賣次數 ===
    st.subheader("🟢 交易次數統計")
    c7, c8 = st.columns(2)
    c7.metric("買進次數", buy_count)
    c8.metric("賣出次數", sell_count)

    # === 匯出結果 CSV ===
    csv = df.to_csv().encode("utf-8")
    st.download_button("⬇️ 下載完整回測結果 CSV", csv, f"{symbol}_LRS_{ma_type}{window}.csv", "text/csv")

    st.success("✅ 回測完成！（已自動抓取前一年暖機資料）")
