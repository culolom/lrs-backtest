# app.py — LRS SMA/EMA 回測系統（美化版績效報表 + 完整可部署）
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
    window = st.slider("均線天數", 10, 200, 200, 10)

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

    # === 績效與風控 ===
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

    loss_streak = (df["Strategy_Return"] < 0).astype(int)
    max_consecutive_loss = loss_streak.groupby(loss_streak.diff().ne(0).cumsum()).transform("size")[loss_streak == 1].max()
    flat_days = (df["Position"] == 0).astype(int)
    max_flat_days = flat_days.groupby(flat_days.diff().ne(0).cumsum()).transform("size")[flat_days == 1].max()

    # === 綜合報表（美化版） ===
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
    .custom-table tr:hover {
      background-color: #fafafa;
    }
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
    <thead>
    <tr><th>指標類別</th><th>指標名稱</th><th>LRS 策略</th><th>Buy & Hold</th></tr>
    </thead>
    <tbody>
    <tr class='section-title'><td colspan='4'>📈 績效指標</td></tr>
    <tr><td></td><td>總報酬</td><td>{final_return_lrs:.2%}</td><td>{final_return_bh:.2%}</td></tr>
    <tr><td></td><td>年化報酬</td><td>{cagr_lrs:.2%}</td><td>{cagr_bh:.2%}</td></tr>
    <tr><td></td><td>最大回撤</td><td>{mdd_lrs:.2%}</td><td>{mdd_bh:.2%}</td></tr>
    <tr><td></td><td>年化波動率</td><td>{vol_lrs:.2%}</td><td>{vol_bh:.2%}</td></tr>
    <tr><td></td><td>夏普值</td><td>{sharpe_lrs:.2f}</td><td>{sharpe_bh:.2f}</td></tr>
    <tr><td></td><td>索提諾值</td><td>{sortino_lrs:.2f}</td><td>{sortino_bh:.2f}</td></tr>

    <tr class='section-title'><td colspan='4'>🛡️ 風控指標</td></tr>
    <tr><td></td><td>最大連續虧損天數</td><td>{int(max_consecutive_loss)} 天</td><td>—</td></tr>
    <tr><td></td><td>最長空倉天數</td><td>{int(max_flat_days)} 天</td><td>—</td></tr>

    <tr class='section-title'><td colspan='4'>💹 交易統計</td></tr>
    <tr><td></td><td>買進次數</td><td>{buy_count}</td><td>—</td></tr>
    <tr><td></td><td>賣出次數</td><td>{sell_count}</td><td>—</td></tr>
    </tbody>
    </table>
    """
    st.markdown(html_table, unsafe_allow_html=True)
 

    # === 年度交易次數圖 ===
    st.markdown("<h3 style='margin-top:2em;'>📊 年度交易次數統計</h3>", unsafe_allow_html=True)
    fig_trade = go.Figure()
    fig_trade.add_trace(go.Bar(x=yearly_trade["年份"], y=yearly_trade["買進次數"], name="買進", marker_color="#27AE60"))
    fig_trade.add_trace(go.Bar(x=yearly_trade["年份"], y=yearly_trade["賣出次數"], name="賣出", marker_color="#E74C3C"))
    fig_trade.update_layout(barmode="group", template="plotly_white", height=400, xaxis_title="年份", yaxis_title="次數")
    st.plotly_chart(fig_trade, use_container_width=True)

    # === 買賣點圖 ===
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(f"{symbol} {ma_type}{window} 買賣訊號", "策略績效對比"), vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="收盤價", line=dict(color="#2E86AB", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA"], mode="lines", name=f"{ma_type}{window}", line=dict(color="#F39C12", width=2)), row=1, col=1)
    if buy_points:
        bx, by = zip(*buy_points)
        fig.add_trace(go.Scatter(x=bx, y=by, mode="markers", name="買進", marker=dict(color="#27AE60", size=9, symbol="triangle-up")), row=1, col=1)
    if sell_points:
        sx, sy = zip(*sell_points)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers", name="賣出", marker=dict(color="#E74C3C", size=9, symbol="x")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_LRS"], mode="lines", name=f"LRS 策略 ({ma_type}{window})", line=dict(color="#16A085", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity_BuyHold"], mode="lines", name="Buy & Hold", line=dict(color="#7F8C8D", width=2, dash="dot")), row=2, col=1)
    fig.update_layout(height=700, template="plotly_white", title=dict(text=f"📈 {symbol} — {ma_type}{window} 回測結果", x=0.0, font=dict(size=26)))
    st.plotly_chart(fig, use_container_width=True)

    # === 年度報酬線圖 ===
    st.markdown("<h3 style='margin-top:2em;'>📆 年度報酬率比較</h3>", unsafe_allow_html=True)
    yearly = df.resample("Y").last()
    yearly["LRS_Annual_Return"] = yearly["Equity_LRS"].pct_change()
    yearly["BH_Annual_Return"] = yearly["Equity_BuyHold"].pct_change()
    if len(yearly) > 1:
        yr = yearly.index.year
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=yr, y=yearly["LRS_Annual_Return"] * 100, mode="lines+markers", name="LRS 年報酬率", line=dict(color="#16A085", width=3)))
        fig_line.add_trace(go.Scatter(x=yr, y=yearly["BH_Annual_Return"] * 100, mode="lines+markers", name="Buy&Hold 年報酬率", line=dict(color="#7F8C8D", width=3, dash="dot")))
        fig_line.update_layout(template="plotly_white", height=400, xaxis_title="年份", yaxis_title="報酬率 (%)", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_line, use_container_width=True)

    # === 月度熱力圖 ===
    st.markdown("<h3 style='margin-top:2em;'>🔥 月度報酬熱力圖 (LRS 策略)</h3>", unsafe_allow_html=True)
    monthly = df["Strategy_Return"].resample("M").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = monthly.to_frame("Monthly_Return")
    monthly_df["Year"] = monthly_df.index.year
    monthly_df["Month"] = monthly_df.index.month
    pivot = monthly_df.pivot(index="Year", columns="Month", values="Monthly_Return") * 100
    pivot = pivot.fillna(0).round(1)
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{m}月" for m in pivot.columns],
        y=pivot.index.astype(str),
        colorscale="RdYlGn",
        zmin=-10, zmax=10,
        text=pivot.round(1).astype(str) + "%",
        texttemplate="%{text}",
        colorbar=dict(title="報酬率 (%)")
    ))
    fig_heat.update_layout(template="plotly_white", height=500, xaxis_title="月份", yaxis_title="年份", title="📊 LRS 策略月度報酬熱力圖")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("<h3 style='margin-top:2em; text-align:left;'>🧾 年度報酬摘要表格 (LRS 策略)</h3>", unsafe_allow_html=True)
    monthly = df["Strategy_Return"].resample("M").apply(lambda x: (1 + x).prod() - 1)
    monthly_df = monthly.to_frame("Monthly_Return")
    monthly_df["Year"] = monthly_df.index.year
    year_summary = []
    for year in sorted(monthly_df["Year"].unique()):
        data = monthly_df[monthly_df["Year"] == year]
        annual_ret = (1 + data["Monthly_Return"]).prod() - 1
        monthly_avg = data["Monthly_Return"].mean()
        win_rate = (data["Monthly_Return"] > 0).mean()
        year_summary.append([year, annual_ret, monthly_avg, win_rate])
    df_summary = pd.DataFrame(year_summary, columns=["年份", "年報酬率", "月平均報酬", "月勝率"])
    avg_year = df_summary["年報酬率"].mean()
    avg_win = df_summary["月勝率"].mean()

    st.markdown("""
    <style>
    table.report-table {
      border-collapse: collapse;
      width: 80%;
      margin: 20px 0;
      font-size: 16px;
      font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      text-align: center;
      box-shadow: 0px 0px 6px rgba(0,0,0,0.08);
      border-radius: 6px;
      overflow: hidden;
    }
    .report-table th {
      background-color: #f0f4ff;
      color: #2c3e50;
      padding: 12px;
      border-bottom: 2px solid #ddd;
    }
    .report-table td {
      padding: 10px;
      border-bottom: 1px solid #eee;
    }
    .report-table tr:hover { background-color: #fafafa; }
    .report-footer {
      background-color: #eaf2ff;
      font-weight: bold;
      text-align: center;
      color: #2c3e50;
    }
    .pos { color: #27ae60; font-weight: 600; }
    .neg { color: #e74c3c; font-weight: 600; }
    .win { color: #2980b9; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

    rows_html = ""
    for _, row in df_summary.iterrows():
        color_class = "pos" if row["年報酬率"] > 0 else "neg"
        rows_html += f"<tr><td>{int(row['年份'])}</td><td class='{color_class}'>{row['年報酬率']:.2%}</td><td>{row['月平均報酬']:.2%}</td><td class='win'>{row['月勝率']*100:.0f}%</td></tr>"
    html_summary = f"""
    <table class='report-table'>
      <thead><tr><th>年份</th><th>年報酬率</th><th>月平均報酬</th><th>月勝率</th></tr></thead>
      <tbody>{rows_html}<tr class='report-footer'><td>平均</td><td>{avg_year:.2%}</td><td>—</td><td>{avg_win*100:.0f}%</td></tr></tbody>
    </table>
    """
    st.markdown(html_summary, unsafe_allow_html=True)

    st.success("✅ 回測完成！")


