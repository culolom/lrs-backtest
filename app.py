# app.py — LRS SMA/EMA 回測系統（含暖機、年/月報酬圖、風控指標）

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

    # === 切掉暖機期間 ===
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

    buy_count = len(buy_points)
    sell_count = len(sell_points)

    # === 年度交易統計 ===
    if buy_points or sell_points:
        buy_years = [d[0].year for d in buy_points]
        sell_years = [d[0].year for d in sell_points]
        buy_series = pd.Series(buy_years).value_counts().sort_index()
        sell_series = pd.Series(sell_years).value_counts().sort_index()
        years = sorted(set(buy_series.index) | set(sell_series.index))
        buy_counts = [buy_series.get(y, 0) for y in years]
        sell_counts = [sell_series.get(y, 0) for y in years]
    else:
        years, buy_counts, sell_counts = [], [], []

    # === 績效計算 ===
    final_return_lrs = df["Equity_LRS"].iloc[-1] - 1
    final_return_bh = df["Equity_BuyHold"].iloc[-1] - 1
    years_len = max((df.index[-1] - df.index[0]).days / 365, 1e-9)
    cagr_lrs = (1 + final_return_lrs) ** (1 / years_len) - 1
    cagr_bh = (1 + final_return_bh) ** (1 / years_len) - 1
    mdd_lrs = 1 - (df["Equity_LRS"] / df["Equity_LRS"].cummax()).min()
    mdd_bh = 1 - (df["Equity_BuyHold"] / df["Equity_BuyHold"].cummax()).min()

    # === 風控指標 ===
    loss_streak = (df["Strategy_Return"] < 0).astype(int)
    max_consecutive_loss = (
        loss_streak.groupby(loss_streak.diff().ne(0).cumsum())
        .transform("size")[loss_streak == 1]
        .max()
    )

    flat_days = (df["Position"] == 0).astype(int)
    max_flat_days = (
        flat_days.groupby(flat_days.diff().ne(0).cumsum())
        .transform("size")[flat_days == 1]
        .max()
    )

    # === 主圖 ===
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
                      title=dict(text=f"📈 {symbol} — {ma_type}{window} 回測",
                                 x=0.0, xanchor="left",
                                 font=dict(size=26, color="#2C3E50", family="Noto Sans TC")),
                      legend=dict(orientation="h", y=-0.25),
                      hovermode="x unified",
                      margin=dict(l=40, r=40, t=80, b=60))
    st.plotly_chart(fig, use_container_width=True)

    # === 回測摘要報表 ===
    st.markdown("## 📄 回測摘要報表")
    col1, col2, col3 = st.columns(3)
    col1.metric("LRS 總報酬", f"{final_return_lrs:.2%}")
    col2.metric("LRS 年化報酬", f"{cagr_lrs:.2%}")
    col3.metric("LRS 最大回撤", f"{mdd_lrs:.2%}")
    col4, col5, col6 = st.columns(3)
    col4.metric("Buy&Hold 總報酬", f"{final_return_bh:.2%}")
    col5.metric("Buy&Hold 年化報酬", f"{cagr_bh:.2%}")
    col6.metric("Buy&Hold 最大回撤", f"{mdd_bh:.2%}")

    # === 新增風控指標 ===
    st.markdown("## 🧱 風險控制分析")
    c1, c2 = st.columns(2)
    c1.metric("最大連續虧損天數", f"{int(max_consecutive_loss) if pd.notna(max_consecutive_loss) else 0} 天")
    c2.metric("最長空倉天數", f"{int(max_flat_days) if pd.notna(max_flat_days) else 0} 天")

    # === 年度交易次數統計 ===
    st.markdown("## 🟢 交易次數統計")
    if years:
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(x=years, y=buy_counts, name="買進次數", marker_color="#27AE60"))
        bar_fig.add_trace(go.Bar(x=years, y=sell_counts, name="賣出次數", marker_color="#E74C3C"))
        bar_fig.update_layout(barmode="group", template="plotly_white",
                              xaxis_title="年份", yaxis_title="次數", height=400,
                              legend=dict(orientation="h", y=1.1))
        st.plotly_chart(bar_fig, use_container_width=True)

    # === 年度報酬率 ===
    st.markdown("## 📈 年度報酬率比較")
    yearly = df.resample("Y").last()
    yearly["LRS_Annual_Return"] = yearly["Equity_LRS"].pct_change()
    yearly["BH_Annual_Return"] = yearly["Equity_BuyHold"].pct_change()
    if len(yearly) > 1:
        yr = yearly.index.year
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(x=yr, y=yearly["LRS_Annual_Return"] * 100,
                                      mode="lines+markers", name="LRS 年報酬率",
                                      line=dict(color="#16A085", width=3)))
        line_fig.add_trace(go.Scatter(x=yr, y=yearly["BH_Annual_Return"] * 100,
                                      mode="lines+markers", name="Buy&Hold 年報酬率",
                                      line=dict(color="#7F8C8D", width=3, dash="dot")))
        line_fig.update_layout(template="plotly_white", xaxis_title="年份",
                               yaxis_title="年報酬率 (%)", height=400,
                               legend=dict(orientation="h", y=1.1))
        st.plotly_chart(line_fig, use_container_width=True)

    # === 月度報酬熱力圖 ===
    st.markdown("## 🔥 月度報酬熱力圖 (LRS 策略)")
    monthly = df["Strategy_Return"].resample("M").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = monthly.to_frame("Monthly_Return")
    monthly_df["Year"] = monthly_df.index.year
    monthly_df["Month"] = monthly_df.index.month
    pivot = monthly_df.pivot(index="Year", columns="Month", values="Monthly_Return") * 100
    pivot = pivot.fillna(0).round(1)
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"{m}月" for m in pivot.columns],
            y=pivot.index.astype(str),
            colorscale="RdYlGn",
            zmin=-10, zmax=10,
            text=pivot.round(1).astype(str) + "%",
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="報酬率 (%)")
        )
    )
    heatmap_fig.update_layout(
        template="plotly_white",
        xaxis_title="月份",
        yaxis_title="年份",
        height=500,
        title="📊 月度報酬熱力圖 (綠=正報酬 / 紅=負報酬)",
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # === 匯出 CSV ===
    csv = df.to_csv().encode("utf-8")
    st.download_button("⬇️ 下載完整回測結果 CSV", csv, f"{symbol}_LRS_{ma_type}{window}.csv", "text/csv")

    st.success("✅ 回測完成！（含年/月報酬圖與風控分析）")
