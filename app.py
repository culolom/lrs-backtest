# app.py — LRS (SMA/EMA + Plotly + 買賣次數 + 年度分佈 + 暖機抓歷史)

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

col4, col5, col6 = st.columns(3)
with col4:
    ma_type = st.selectbox("均線種類", ["SMA", "EMA"])
with col5:
    window = st.slider("均線天數", 50, 300, 200, 10)
with col6:
    warmup_days = st.slider("前置抓取天數（暖機）", 200, 730, 365, 5)  # 預設抓 1 年

# === 按下按鈕後回測 ===
if st.button("開始回測 🚀"):
    # 1) 先把 start 往前挪，確保 MA 暖機資料充足
    #    取 max(使用者設定的 warmup_days, 2*window) 比較穩
    warmup_needed = max(warmup_days, 2 * window)
    start_early = pd.to_datetime(start) - pd.Timedelta(days=warmup_needed)

    with st.spinner(f"資料下載中…（含暖機 {warmup_needed} 天）"):
        # 注意：yfinance 會自動 auto_adjust=True（分割/股利調整），這是正常的
        df_raw = yf.download(symbol, start=start_early, end=end)
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

    if df_raw.empty or "Close" not in df_raw:
        st.error("下載不到資料，請換一個代號或日期區間。")
        st.stop()

    # 2) 計算 MA（用「暖機後的完整資料」）
    df = df_raw.copy()
    if ma_type == "SMA":
        df["MA"] = df["Close"].rolling(window=window).mean()
    else:
        df["MA"] = df["Close"].ewm(span=window, adjust=False).mean()

    # 3) 計算訊號／績效（仍用完整資料，避免邊界影響）
    df["Signal"] = np.where(df["Close"] > df["MA"], 1, 0)
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Return"] * df["Signal"]
    df["Equity_LRS"] = (1 + df["Strategy_Return"]).cumprod()
    df["Equity_BuyHold"] = (1 + df["Return"]).cumprod()

    # 4) 真正「展示 & 評分」時，再裁回使用者選的開始日（避免暖機期干擾圖表與統計）
    df = df.loc[pd.to_datetime(start): pd.to_datetime(end)]

    if df.empty:
        st.error("暖機後裁切為空，請調整日期。")
        st.stop()

    # === 計算績效指標 ===
    final_return_lrs = df["Equity_LRS"].iloc[-1] - 1
    final_return_bh = df["Equity_BuyHold"].iloc[-1] - 1
    years = max((df.index[-1] - df.index[0]).days / 365, 1e-9)  # 避免除以 0
    cagr_lrs = (1 + final_return_lrs) ** (1 / years) - 1
    cagr_bh = (1 + final_return_bh) ** (1 / years) - 1
    mdd_lrs = 1 - (df["Equity_LRS"] / df["Equity_LRS"].cummax()).min()
    mdd_bh = 1 - (df["Equity_BuyHold"] / df["Equity_BuyHold"].cummax()).min()

    # === 建立買賣點（用裁切後的資料）
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

    # === 年度交易次數統計 ===
    buy_years = pd.Series([b[0].year for b in buy_points], dtype="Int64") if buy_points else pd.Series(dtype="Int64")
    sell_years = pd.Series([s[0].year for s in sell_points], dtype="Int64") if sell_points else pd.Series(dtype="Int64")
    years_all = sorted(set(buy_years.dropna().tolist() + sell_years.dropna().tolist()))
    trade_df = pd.DataFrame({"Year": years_all})
    trade_df["Buy"] = [int((buy_years == y).sum()) for y in years_all] if years_all else []
    trade_df["Sell"] = [int((sell_years == y).sum()) for y in years_all] if years_all else []

    # === Plotly 主圖 ===
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=(f"{symbol} {ma_type}{window} 買賣訊號", "策略績效對比"),
        vertical_spacing=0.1,
    )

    # 價格走勢 + MA
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines",
                             name="收盤價", line=dict(color="#2E86AB", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA"], mode="lines",
                             name=f"{ma_type}{window}", line=dict(color="#F39C12", width=2)), row=1, col=1)

    # 買賣點
    if buy_points:
        bx, by = zip(*buy_points)
        fig.add_trace(go.Scatter(x=bx, y=by, mode="markers", name="買進",
                                 marker=dict(color="#27AE60", size=9, symbol="triangle-up")), row=1, col=1)
    if sell_points:
        sx, sy = zip(*sell_points)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers", name="賣出",
                                 marker=dict(color="#E74C3C", size=9, symbol="x")), row=1, col=1)

    # 策略績效
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_LRS"], mode="lines",
                             name=f"LRS 策略 ({ma_type}{window})", line=dict(color="#16A085", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_BuyHold"], mode="lines",
                             name="Buy & Hold", line=dict(color="#7F8C8D", width=2, dash="dot")), row=2, col=1)

    fig.update_layout(
        height=700,
        template="plotly_white",
        title=dict(text=f"📈 {symbol} — {ma_type}{window}（暖機 {warmup_needed} 天）", x=0.5, font=dict(size=20)),
        legend=dict(orientation="h", y=-0.25),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # === 總體績效 ===
    st.subheader("📊 回測績效摘要")
    c1, c2, c3 = st.columns(3)
    c1.metric("LRS 總報酬", f"{final_return_lrs:.2%}")
    c2.metric("LRS 年化報酬", f"{cagr_lrs:.2%}")
    c3.metric("LRS 最大回撤", f"{mdd_lrs:.2%}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Buy&Hold 總報酬", f"{final_return_bh:.2%}")
    c5.metric("Buy&Hold 年化報酬", f"{cagr_bh:.2%}")
    c6.metric("Buy&Hold 最大回撤", f"{mdd_bh:.2%}")

    # === 交易次數 ===
    st.subheader("🟢 交易次數統計")
    c7, c8 = st.columns(2)
    c7.metric("買進次數", buy_count)
    c8.metric("賣出次數", sell_count)

    # === 年度交易次數分佈圖 ===
    if not trade_df.empty:
        fig_trade = go.Figure()
        fig_trade.add_bar(x=trade_df["Year"], y=trade_df["Buy"], name="買進次數", marker_color="#27AE60")
        fig_trade.add_bar(x=trade_df["Year"], y=trade_df["Sell"], name="賣出次數", marker_color="#E74C3C")
        fig_trade.update_layout(
            barmode="group",
            template="plotly_white",
            title="📅 每年交易次數分佈",
            xaxis_title="年份", yaxis_title="次數", height=400,
        )
        st.plotly_chart(fig_trade, use_container_width=True)

    # === 匯出 CSV（裁切後區間） ===
    csv = df.to_csv().encode("utf-8")
    st.download_button("⬇️ 下載完整回測結果 CSV", csv, f"{symbol}_LRS_{ma_type}{window}_warm{warmup_needed}.csv", "text/csv")

    st.success("✅ 回測完成！")
