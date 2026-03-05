import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === ✨ Mac 中文顯示設定 ===
plt.rcParams['font.sans-serif'] = ['Heiti TC'] # 設定使用 Mac 內建的繁體中文字體
plt.rcParams['axes.unicode_minus'] = False     # 解決座標軸負號顯示問題

# 1. 讀取 CSV
try:
    df = pd.read_csv('stock_ai_deep_analysis.csv')
except FileNotFoundError:
    print("❌ 找不到 CSV 檔案，請先執行 git pull 同步資料！")
    exit()

# 2. 資料預處理
df['抓取時間'] = pd.to_datetime(df['抓取時間'])
df = df.sort_values('抓取時間')

# 3. 建立畫布
fig, ax1 = plt.subplots(figsize=(12, 6))

# 4. 畫出股價走勢
ax1.set_xlabel('時間')
ax1.set_ylabel('股價', color='tab:blue')
ax1.plot(df['抓取時間'], df['當時股價'], color='tab:blue', marker='o', label='台積電股價', alpha=0.5)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 5. 標註 AI 情緒 (修正了之前的「股軍」打字錯誤)
pos_news = df[df['AI 情緒標籤'] == 'Positive']
neg_news = df[df['AI 情緒標籤'] == 'Negative']

# 用紅點表示 AI 看多，綠點表示 AI 看空
ax1.scatter(pos_news['抓取時間'], pos_news['當時股價'], color='red', s=100, label='AI 正面評價', zorder=5)
ax1.scatter(neg_news['抓取時間'], neg_news['當時股價'], color='green', s=100, label='AI 負面評價', zorder=5)

# 6. 美化圖表
plt.title('台積電 AI 情緒與股價關聯分析')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()