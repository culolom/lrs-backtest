import matplotlib.font_manager as fm
import matplotlib

# === 字型設定 ===
font_path = "./NotoSansTC-Bold.ttf"  # 注意：檔名要完全相同（含大小寫）
fm.fontManager.addfont(font_path)
matplotlib.rcParams["font.family"] = "Noto Sans TC"
matplotlib.rcParams["axes.unicode_minus"] = False
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib

# === 中文字型設定 ===
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="布林通道缺點事件分析", page_icon="📊", layout="wide")
st.title("📈 布林通道缺點事件分析")

# === 使用者輸入 ===
symbol = st.text_input("輸入代號（例如 TQQQ, SPY, 00631L.TW）", "TQQQ")
start = st.date_input("開始日期", pd.to_datetime("2015-01-01"))
end = st.date_input("結束日期", pd.to_datetime("2025-01-01"))

if st.button("開始分析 🚀"):
    with st.spinner("資料下載中..."):
        df = yf.download(symbol, start=start, end=end)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        st.success(f"✅ 已下載 {len(df)} 筆 {symbol} 歷史資料")

    # === 計算布林通道 ===
    df['SMA'] = df['Close'].rolling(200).mean()
    df['STD'] = df['Close'].rolling(200).std()
    df['Upper'] = df['SMA'] + 2 * df['STD']
    df['Lower'] = df['SMA'] - 2 * df['STD']

    close = df['Close'].to_numpy()
    upper = df['Upper'].to_numpy()
    lower = df['Lower'].to_numpy()

    df['Above'] = close > upper
    df['Below'] = close < lower

    # === 缺點事件統計 ===
    def count_defects(signal):
        count = 0
        active = False
        result = []
        for val in signal:
            if not active and val:
                active = True
                count += 1
            elif active and not val:
                active = False
            result.append(count)
        return result

    df['Up_Defect_Count'] = count_defects(df['Above'])
    df['Down_Defect_Count'] = count_defects(df['Below'])
    df['Total_Defects'] = df['Up_Defect_Count'] + df['Down_Defect_Count']

    # === 趨勢結構分析 ===
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['Below200'] = df['Close'] < df['SMA200']
    df['Fatal'] = (df['Up_Defect_Count'] >= 4) | (df['Down_Defect_Count'] >= 4)

    # === 繪圖 ===
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df['Close'], label='Close', color='blue')
    ax.plot(df.index, df['SMA'], label='SMA200', color='orange', alpha=0.8)
    ax.plot(df.index, df['Upper'], '--', color='grey', alpha=0.6)
    ax.plot(df.index, df['Lower'], '--', color='grey', alpha=0.6)
    ax.scatter(df.index[df['Fatal']], df['Close'][df['Fatal']],
               color='red', marker='x', label='致命缺點 (≥4次)')
    ax.scatter(df.index[df['Below200']], df['Close'][df['Below200']],
               color='black', marker='v', label='跌破200SMA')
    ax.set_title(f"{symbol} 布林通道缺點事件分析")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # === 統計摘要 ===
    fatal_times = int(df['Fatal'].sum())
    below200_times = int(df['Below200'].sum())

    st.subheader("📊 統計摘要")
    st.write(f"上漲缺點事件最大次數：{df['Up_Defect_Count'].max()}")
    st.write(f"下跌缺點事件最大次數：{df['Down_Defect_Count'].max()}")
    st.write(f"致命缺點（≥4次）出現次數：{fatal_times}")
    st.write(f"跌破200SMA 次數：{below200_times}")

    st.success("✅ 分析完成！")




