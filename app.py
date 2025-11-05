import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常顯示負號

print("📈 開始下載資料中...")

symbol = "TQQQ"
start = "2015-01-01"
end = "2025-01-01"

# 下載資料
df = yf.download(symbol, start=start, end=end)
print(f"✅ 已下載 {len(df)} 筆 {symbol} 歷史資料")

# 若是 MultiIndex 欄位（新版 yfinance 常出現）
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# === 計算布林通道 ===
df['SMA'] = df['Close'].rolling(200).mean()
df['STD'] = df['Close'].rolling(200).std()
df['Upper'] = df['SMA'] + 2 * df['STD']
df['Lower'] = df['SMA'] - 2 * df['STD']

# 確保用 numpy 進行 1D 比較
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
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Close', color='blue')
plt.plot(df.index, df['SMA'], label='SMA20', color='orange', alpha=0.8)
plt.plot(df.index, df['Upper'], '--', color='grey', alpha=0.6)
plt.plot(df.index, df['Lower'], '--', color='grey', alpha=0.6)

plt.scatter(df.index[df['Fatal']], df['Close'][df['Fatal']],
            color='red', marker='x', label='致命缺點 (≥4次)')
plt.scatter(df.index[df['Below200']], df['Close'][df['Below200']],
            color='black', marker='v', label='跌破200SMA')

plt.title(f"{symbol} 布林通道缺點事件分析")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# === 統計摘要 ===
fatal_times = df['Fatal'].sum()
below200_times = df['Below200'].sum()

print("\n📊 統計摘要")
print(f"上漲缺點事件最大次數：{df['Up_Defect_Count'].max()}")
print(f"下跌缺點事件最大次數：{df['Down_Defect_Count'].max()}")
print(f"致命缺點（≥4次）出現次數：{fatal_times}")
print(f"跌破200SMA 次數：{below200_times}")
print("\n✅ 分析完成！圖表已顯示。")
