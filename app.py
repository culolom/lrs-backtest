# app.py — LRS 回測系統（最終版：含本金模擬 + 互動圖 + 美化報表）
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

col4, col5, col6 = st.columns(3)
with col4:
    ma_type = st.selectbox("均線種類", ["SMA", "EMA"])
with col5:
    window = st.slider("均線天數", 50, 200, 200, 10)
with col6:
    initial_capital = st.number_input("投入本金（元）", 1000, 1_000_000, 10000, step=1000)

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

    # 第一筆強制持有（確保公平起點）
    df.loc[df.index[0], "Position"] = 1

    # === 策略報酬 ===
    df["Strategy_Return"] = df["Return"] * df["Position"]
    df["Equity_LRS"] = (1 + df["Strategy_Return"]).cumprod()
    df["Equity_BuyHold"] = (1 + df["Return"]).cumprod()
    df = df.loc[pd.to_datetime(start): pd.to_datetime(end)].copy()
    df["Equity_LRS"] /= df["Equity_LRS"].iloc[0]
    df["Equity_BuyHold"] /= df["Equity_BuyHold"].iloc[0]

    # === 本金模擬 ===
    df["LRS_Capital"] = df["Equity_LRS"] * initial_capital
    df["BH_Capital"] = df["Equity_BuyHold"] * initial_capital

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

    # === 績效 ===
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

    # === 最終資產 ===
    equity_lrs_final = df["LRS_Capital"].iloc[-1]
    equity_bh_final = df["BH_Capital"].iloc[-1]

    # === Plotly 圖表 ===
    st.markdown("<h2 style='margin-top:1em;'>📈 策略與績效視覺化</h2>", unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("收盤價與均線（含買賣點）", "策略績效曲線"))

    # 主圖
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="收盤價", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA"], name=f"{ma_type}{window}", line=dict(color="orange")), row=1, col=1)
    if buy_points:
        bx, by = zip(*buy_points)
        fig.add_trace(go.Scatter(x=bx, y=by, mode="markers", name="買進", marker=dict(color="green", symbol="triangle-up", size=8)), row=1, col=1)
    if sell_points:
        sx, sy = zip(*sell_points)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers", name="賣出", marker=dict(color="red", symbol="x", size=8)), row=1, col=1)

    # 績效圖
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_LRS"], name="LRS 策略", line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_BuyHold"], name="Buy & Hold", line=dict(color="gray", dash="dot")), row=2, col=1)
    fig.update_layout(height=800, showlegend=True, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # === 綜合績效表格 ===
    st.markdown("""
    <h2 style='margin-top:1.5em; text-align:left;'>📊 綜合回測績效報表</h2>
    <style>
    table.custom-table {
      border-collapse: collapse;
      width: 95%;
      margin: 25px 0;
      font-size: 16px;
      text-align: center;
      font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      box-shadow: 0px 0px 6px rgba(0,0,0,0.1);
    }
    .custom-table th {
      background-color: #f0f4ff;
      padding: 12px;
      border-bottom: 2px solid #ddd;
      font-weight: bold;
      color: #2c3e50;
    }
    .custom-table td {
      padding: 10px;
      border-bottom: 1px solid #eee;
      color: #2c3e50;
    }
    .custom-table tr:hover { background-color: #fafafa; }
    .section-title {
      background-color: #eaf2ff;
      font-weight: bold;
      text-align: left;
      padding-left: 10px;
      color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

    html_table = f"""
    <table class='custom-table'>
    <thead><tr><th>指標名稱</th><th>LRS 策略</th><th>Buy & Hold</th></tr></thead>
    <tbody>
    <tr><td>最終資產</td><td>{equity_lrs_final:,.0f} 元</td><td>{equity_bh_final:,.0f} 元</td></tr>
    <tr><td>總報酬</td><td>{final_return_lrs:.2%}</td><td>{final_return_bh:.2%}</td></tr>
    <tr><td>年化報酬</td><td>{cagr_lrs:.2%}</td><td>{cagr_bh:.2%}</td></tr>
    <tr><td>最大回撤</td><td>{mdd_lrs:.2%}</td><td>{mdd_bh:.2%}</td></tr>
    <tr><td>年化波動率</td><td>{vol_lrs:.2%}</td><td>{vol_bh:.2%}</td></tr>
    <tr><td>夏普值</td><td>{sharpe_lrs:.2f}</td><td>{sharpe_bh:.2f}</td></tr>
    <tr><td>索提諾值</td><td>{sortino_lrs:.2f}</td><td>{sortino_bh:.2f}</td></tr>
    <tr class='section-title'><td colspan='3'>💹 交易統計</td></tr>
    <tr><td>買進次數</td><td>{buy_count}</td><td>—</td></tr>
    <tr><td>賣出次數</td><td>{sell_count}</td><td>—</td></tr>
    </tbody></table>
    """
    st.markdown(html_table, unsafe_allow_html=True)

    st.success("✅ 回測完成！")
