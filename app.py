# app.py — LRS 股債切換回測系統（含配息公債 + 自動台股辨識）

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
st.set_page_config(page_title="LRS 股債切換回測系統", page_icon="📊", layout="wide")
st.markdown("<h1 style='margin-bottom:0.5em;'>📊 Leverage Rotation Strategy — 股債切換回測系統</h1>", unsafe_allow_html=True)

# === 自動補 .TW 的函式 ===
def normalize_symbol(symbol):
    s = symbol.strip().upper()
    if s.isdigit() or (not "." in s and (s.startswith("00") or s.startswith("23") or s.startswith("008"))):
        s += ".TW"
    return s

# === 使用者輸入區 ===
col1, col2 = st.columns(2)
with col1:
    stock_symbol_raw = st.text_input("股票代號（例：0050, QQQ, SPY, 00631L.TW）", "0050")
with col2:
    bond_symbol_raw = st.text_input("債券代號（例：TLT, IEF, 00679B.TW）", "00679B.TW")

stock_symbol = normalize_symbol(stock_symbol_raw)
bond_symbol = normalize_symbol(bond_symbol_raw)

col3, col4, col5 = st.columns(3)
with col3:
    start = st.date_input("開始日期", pd.to_datetime("2013-01-01"))
with col4:
    end = st.date_input("結束日期", pd.to_datetime("2025-01-01"))
with col5:
    initial_capital = st.number_input("投入本金（元）", 1000, 2_000_000, 10000, step=1000)

window = st.slider("均線天數", 50, 200, 200, step=10)

# === 主程式 ===
if st.button("開始回測 🚀"):

    start_early = pd.to_datetime(start) - pd.Timedelta(days=365)
    with st.spinner("下載股票與債券資料中（自動暖機一年）..."):
        df_stock = yf.download(stock_symbol, start=start_early, end=end)
        df_bond = yf.download(bond_symbol, start=start_early, end=end)

    if df_stock.empty or df_bond.empty:
        st.error("⚠️ 無法下載資料，請確認代號或時間。")
        st.stop()

    # 使用含配息的調整後收盤價
    df_stock["Price"] = df_stock["Adj Close"]
    df_bond["Price"] = df_bond["Adj Close"]

    # 均線與訊號
    df_stock["SMA200"] = df_stock["Price"].rolling(window=window).mean()
    df_stock["Signal"] = np.where(df_stock["Price"] > df_stock["SMA200"], 1, 0)

    # 對齊日期
    common_index = df_stock.index.intersection(df_bond.index)
    df_stock = df_stock.loc[common_index]
    df_bond = df_bond.loc[common_index]

    # 日報酬率
    stock_ret = df_stock["Price"].pct_change().fillna(0)
    bond_ret = df_bond["Price"].pct_change().fillna(0)

    # 策略報酬（股 > 均線 投股票，反之投債）
    strategy_ret = np.where(df_stock["Signal"] == 1, stock_ret, bond_ret)
    df_stock["Strategy_Return"] = strategy_ret
    df_stock["Return"] = stock_ret
    df_stock["Bond_Return"] = bond_ret

    # 累積曲線
    df_stock["Equity_LRS"] = (1 + df_stock["Strategy_Return"]).cumprod()
    df_stock["Equity_BuyHold"] = (1 + df_stock["Return"]).cumprod()
    df_stock["Equity_Bond"] = (1 + df_stock["Bond_Return"]).cumprod()

    df_stock = df_stock.loc[pd.to_datetime(start): pd.to_datetime(end)]
    df_stock["LRS_Capital"] = df_stock["Equity_LRS"] * initial_capital
    df_stock["BH_Capital"] = df_stock["Equity_BuyHold"] * initial_capital

    # === 買賣點 ===
    buy_points = [(df_stock.index[i], df_stock["Price"].iloc[i]) for i in range(1, len(df_stock)) if df_stock["Signal"].iloc[i] == 1 and df_stock["Signal"].iloc[i-1] == 0]
    sell_points = [(df_stock.index[i], df_stock["Price"].iloc[i]) for i in range(1, len(df_stock)) if df_stock["Signal"].iloc[i] == 0 and df_stock["Signal"].iloc[i-1] == 1]
    buy_count, sell_count = len(buy_points), len(sell_points)

    # === 指標計算 ===
    years = (df_stock.index[-1] - df_stock.index[0]).days / 365
    def metrics(series):
        r = series.dropna()
        mean = r.mean()
        std = r.std()
        downside = r[r < 0].std()
        vol = std * np.sqrt(252)
        sharpe = (mean / std) * np.sqrt(252) if std > 0 else np.nan
        sortino = (mean / downside) * np.sqrt(252) if downside > 0 else np.nan
        return vol, sharpe, sortino

    def summary(eq):
        total = eq.iloc[-1] - 1
        cagr = (1 + total) ** (1 / years) - 1
        mdd = 1 - (eq / eq.cummax()).min()
        return total, cagr, mdd

    final_lrs, cagr_lrs, mdd_lrs = summary(df_stock["Equity_LRS"])
    final_bh, cagr_bh, mdd_bh = summary(df_stock["Equity_BuyHold"])
    vol_lrs, sharpe_lrs, sortino_lrs = metrics(df_stock["Strategy_Return"])
    vol_bh, sharpe_bh, sortino_bh = metrics(df_stock["Return"])

    # === 圖表 ===
    st.markdown("<h2>📊 策略績效視覺化</h2>", unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("收盤價與均線", "資金曲線"))
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock["Price"], name="股價", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock["SMA200"], name="SMA200", line=dict(color="orange")), row=1, col=1)
    if buy_points:
        bx, by = zip(*buy_points)
        fig.add_trace(go.Scatter(x=bx, y=by, mode="markers", name="買進", marker=dict(color="green", symbol="triangle-up", size=8)), row=1, col=1)
    if sell_points:
        sx, sy = zip(*sell_points)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers", name="賣出", marker=dict(color="red", symbol="x", size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock["Equity_LRS"], name="LRS 策略", line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock["Equity_BuyHold"], name="Buy & Hold", line=dict(color="gray", dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock["Equity_Bond"], name="Bond", line=dict(color="purple", dash="dot")), row=2, col=1)
    fig.update_layout(height=800, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # === CSS 美化表格 ===
    st.markdown("""
    <style>
    .custom-table {
        width:100%; border-collapse:collapse; margin-top:1.2em;
        font-family:"Noto Sans TC"; box-shadow:0 3px 8px rgba(0,0,0,0.05);
    }
    .custom-table th {
        background:#f5f6fa; padding:12px; font-weight:700; border-bottom:2px solid #ddd;
    }
    .custom-table td {
        text-align:center; padding:10px; border-bottom:1px solid #eee; font-size:15px;
    }
    .section-title td {
        background:#eef4ff; color:#1a237e; font-weight:700; text-align:left;
    }
    </style>
    """, unsafe_allow_html=True)

    html_table = f"""
    <table class='custom-table'>
    <thead><tr><th>指標名稱</th><th>LRS 策略（股債切換）</th><th>Buy & Hold（股票）</th></tr></thead>
    <tbody>
    <tr><td>最終資產</td><td>{df_stock['LRS_Capital'].iloc[-1]:,.0f} 元</td><td>{df_stock['BH_Capital'].iloc[-1]:,.0f} 元</td></tr>
    <tr><td>總報酬</td><td>{final_lrs:.2%}</td><td>{final_bh:.2%}</td></tr>
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
    st.success("✅ 回測完成！（LRS 策略已支援股債切換）")
