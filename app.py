# app.py — Leverage Rotation Strategy (SMA200版, Streamlit 互動版)

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.font_manager as fm
import matplotlib

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
st.set_page_config(page_title="LRS SMA200 回測系統", page_icon="📈", layout="wide")
st.title("📊 Leverage Rotation Strategy — SMA200 基本版")

# === 使用者輸入區 ===
col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.text_input("輸入代號（例：00631L.TW, QQQ, SPXL, BTC-USD）", "00631L.TW")
with col2:
    start = st.date_input("開始日期", pd.to_datetime("2023-01-01"))
with col3:
    end = st.date_input("結束日期", pd.to_datetime("2025-01-01"))

if st.button("開始回測 🚀"):
    with st.spinner("資料下載中..."):
        df = yf.download(symbol, start=start, end=end)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        st.success(f"✅ 已下載 {len(df)} 筆 {symbol} 歷史資料")

    # === 計算 SMA200 ===
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    df["Signal"] = np.where(df["Close"] > df["SMA200"], 1, 0)

    # === 計算每日報酬與策略報酬 ===
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Strategy_Return"] = df["Return"] * df["Signal"]
    df["Equity_LRS"] = (1 + df["Strategy_Return"]).cumprod()
    df["Equity_BuyHold"] = (1 + df["Return"]).cumprod()

    # === 計算績效指標 ===
    final_return_lrs = df["Equity_LRS"].iloc[-1] - 1
    final_return_bh = df["Equity_BuyHold"].iloc[-1] - 1
    years = (df.index[-1] - df.index[0]).days / 365
    cagr_lrs = (1 + final_return_lrs) ** (1 / years) - 1
    cagr_bh = (1 + final_return_bh) ** (1 / years) - 1
    mdd_lrs = 1 - (df["Equity_LRS"] / df["Equity_LRS"].cummax()).min()
    mdd_bh = 1 - (df["Equity_BuyHold"] / df["Equity_BuyHold"].cummax()).min()

    # === 建立買賣點 ===
    buy_points, sell_points = [], []
    prev_signal = 0
    for i in range(len(df)):
        signal = df["Signal"].iloc[i]
        price = df["Close"].iloc[i]
        if signal == 1 and prev_signal == 0:
            buy_points.append((df.index[i], price))
        elif signal == 0 and prev_signal == 1:
            sell_points.append((df.index[i], price))
        prev_signal = signal

    # === 圖表 ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    # (1) 價格走勢 + 買賣點
    ax1.plot(df.index, df["Close"], label="收盤價", color="blue")
    ax1.plot(df.index, df["SMA200"], label="SMA200", color="orange")
    if buy_points:
        bx, by = zip(*buy_points)
        ax1.scatter(bx, by, color="green", marker="^", s=80, label="買進（突破SMA200）")
    if sell_points:
        sx, sy = zip(*sell_points)
        ax1.scatter(sx, sy, color="red", marker="x", s=70, label="賣出（跌破SMA200）")
    ax1.legend()
    ax1.set_title(f"{symbol} LRS 基本版（SMA200）：突破買進、跌破賣出")

    # (2) 策略績效對比
    ax2.plot(df.index, df["Equity_LRS"], color="green", label="LRS 策略 (SMA200)")
    ax2.plot(df.index, df["Equity_BuyHold"], color="grey", linestyle="--", label="Buy & Hold")
    ax2.legend()
    ax2.set_title("策略績效曲線對比")

    text = (
        f"LRS(SMA200) 總報酬: {final_return_lrs:.2%}\n"
        f"LRS(SMA200) 年化報酬(CAGR): {cagr_lrs:.2%}\n"
        f"LRS(SMA200) 最大回撤(MDD): {mdd_lrs:.2%}\n"
        f"Buy&Hold 總報酬: {final_return_bh:.2%}\n"
        f"Buy&Hold 年化報酬(CAGR): {cagr_bh:.2%}\n"
        f"Buy&Hold 最大回撤(MDD): {mdd_bh:.2%}"
    )
    ax2.text(df.index[int(len(df) * 0.02)], df["Equity_LRS"].max() * 0.7, text,
             fontsize=10, bbox=dict(facecolor="white", alpha=0.6))
    plt.tight_layout()
    st.pyplot(fig)

    # === 顯示回測結果 ===
    st.subheader("📊 回測績效摘要")
    col1, col2, col3 = st.columns(3)
    col1.metric("LRS 總報酬", f"{final_return_lrs:.2%}")
    col2.metric("LRS 年化報酬", f"{cagr_lrs:.2%}")
    col3.metric("LRS 最大回撤", f"{mdd_lrs:.2%}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Buy&Hold 總報酬", f"{final_return_bh:.2%}")
    col5.metric("Buy&Hold 年化報酬", f"{cagr_bh:.2%}")
    col6.metric("Buy&Hold 最大回撤", f"{mdd_bh:.2%}")

    # === 匯出結果 CSV ===
    csv = df.to_csv().encode("utf-8")
    st.download_button("⬇️ 下載完整回測結果 CSV", csv, f"{symbol}_LRS_SMA200.csv", "text/csv")

    st.success("✅ 回測完成！")
