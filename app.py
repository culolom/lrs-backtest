# app.py — LRS SMA/EMA 回測系統（整合報表版）

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
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# === Streamlit 頁面設定 ===
st.set_page_config(page_title="LRS 回測系統", page_icon="📈", layout="wide")
st.title("📊 Leverage Rotation Strategy — SMA / EMA 回測系統")

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

    # === 累積報酬 ===
    df["Equity_LRS"] = (1 + df["Strategy_Return"]).cumprod()
    df["Equity_BuyHold"] = (1 + df["Return"]).cumprod()

    # === 切掉暖機區間 ===
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

    # === 績效計算 ===
    final_return_lrs = df["Equity_LRS"].iloc[-1] - 1
    final_return_bh = df["Equity_BuyHold"].iloc[-1] - 1
    years_len = max((df.index[-1] - df.index[0]).days / 365, 1e-9)
    cagr_lrs = (1 + final_return_lrs) ** (1 / years_len) - 1
    cagr_bh = (1 + final_return_bh) ** (1 / years_len) - 1
    mdd_lrs = 1 - (df["Equity_LRS"] / df["Equity_LRS"].cummax()).min()
    mdd_bh = 1 - (df["Equity_BuyHold"] / df["Equity_BuyHold"].cummax()).min()

    # === 策略穩定性：LRS & BuyHold 對照 ===
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

    # === 📊 綜合回測績效報表 ===
    st.markdown("## 📊 綜合回測績效報表 (LRS vs Buy&Hold)")

    summary_data = {
        "指標": [
            "總報酬",
            "年化報酬",
            "最大回撤",
            "年化波動率",
            "夏普值",
            "索提諾值"
        ],
        "LRS": [
            f"{final_return_lrs:.2%}",
            f"{cagr_lrs:.2%}",
            f"{mdd_lrs:.2%}",
            f"{vol_lrs:.2%}",
            f"{sharpe_lrs:.2f}",
            f"{sortino_lrs:.2f}"
        ],
        "Buy&Hold": [
            f"{final_return_bh:.2%}",
            f"{cagr_bh:.2%}",
            f"{mdd_bh:.2%}",
            f"{vol_bh:.2%}",
            f"{sharpe_bh:.2f}",
            f"{sortino_bh:.2f}"
        ]
    }

    def compare_metrics(lrs, bh, higher_better=True):
        try:
            lrs_val = float(lrs.strip('%'))
            bh_val = float(bh.strip('%'))
            return "勝" if (lrs_val > bh_val if higher_better else lrs_val < bh_val) else "敗"
        except:
            try:
                return "勝" if float(lrs) > float(bh) else "敗"
            except:
                return "—"

    comparison = []
    for i, k in enumerate(summary_data["指標"]):
        if k in ["最大回撤"]:
            comparison.append(compare_metrics(summary_data["LRS"][i], summary_data["Buy&Hold"][i], higher_better=False))
        else:
            comparison.append(compare_metrics(summary_data["LRS"][i], summary_data["Buy&Hold"][i], higher_better=True))
    summary_data["評比"] = comparison

    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

    # === 📈 主圖 ===
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(f"{symbol} {ma_type}{window} 買賣訊號", "策略績效對比"),
                        vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines",
                             name="收盤價", line=dict(color="#2E86AB", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA"], mode="lines",
                             name=f"{ma_type}{window}", line=dict(color="#F39C12", width=2)), row=1, col=1)
    if buy_points:
        bx, by = zip(*buy_points)
        fig.add_trace(go.Scatter(x=bx, y=by, mode="markers", name="買進",
                                 marker=dict(color="#27AE60", size=9, symbol="triangle-up")), row=1, col=1)
    if sell_points:
        sx, sy = zip(*sell_points)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers", name="賣出",
                                 marker=dict(color="#E74C3C", size=9, symbol="x")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_LRS"], mode="lines",
                             name=f"LRS 策略 ({ma_type}{window})", line=dict(color="#16A085", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_BuyHold"], mode="lines",
                             name="Buy & Hold", line=dict(color="#7F8C8D", width=2, dash="dot")), row=2, col=1)
    fig.update_layout(height=700, template="plotly_white",
                      title=dict(text=f"📈 {symbol} — {ma_type}{window} 回測", x=0.0, xanchor="left", font=dict(size=26)))
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ 回測完成！")
