import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ==========================================
# 1. 配置與模型初始化
# ==========================================
# 使用你測試成功、最準確的中文財經模型
MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
TARGET_STOCK = "2330.TW"  # 台積電
CSV_FILE = "stock_ai_master_data.csv"
KEYWORDS = ["台積電", "TSMC", "AI", "半導體", "輝達", "營收", "飆", "漲", "跌", "財報"]

print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在啟動 AI 系統 ...")

def load_ai_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f" 模型載入失敗: {e}")
        return None

# ==========================================
# 2. 核心功能函數
# ==========================================

def get_stock_data(stock_id):
    """獲取精確的漲跌幅（對比昨日收盤價）"""
    try:
        stock = yf.Ticker(stock_id)
        # 抓取最近兩天的資料，確保能拿到昨天收盤和今天現價
        df = stock.history(period="2d")
        
        if len(df) >= 2:
            prev_close = df['Close'].iloc[-2]  # 昨天的收盤價
            current_price = df['Close'].iloc[-1]  # 今天的最新價
            
            # 這是市場通用的漲跌幅公式
            change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)
            return round(current_price, 2), change_pct
        else:
            # 如果是剛開盤或資料不足，回傳 info 裡的預設值
            current_price = stock.info.get('regularMarketPrice', 0)
            prev_close = stock.info.get('previousClose', 1)
            change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)
            return round(current_price, 2), change_pct
    except Exception as e:
        print(f"股價抓取異常: {e}")
        return 0, 0

def generate_visuals(df):
    """生成情緒分佈與信心圖表"""
    if df.empty: return
    
    # 設置繪圖字體 (Mac 預設繁體字體)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
    
    # 1. 情緒分佈圖
    plt.figure(figsize=(10, 5))
    counts = df['AI 情緒標籤'].value_counts()
    counts.plot(kind='bar', color=['#2ecc71', '#e74c3c', '#95a5a6'])
    plt.title(f'市場情緒分佈 - {datetime.now().strftime("%Y-%m-%d")}')
    plt.ylabel('新聞篇數')
    plt.savefig('sentiment_report.png')
    plt.close()
    print(" 圖表已更新:sentiment_report.png")

# ==========================================
# 3. 主程式邏輯
# ==========================================

def main():
    nlp_pipe = load_ai_model()
    if not nlp_pipe: return

    # A. 抓取股價
    price, change = get_stock_data(TARGET_STOCK)
    
    # B. 爬取新聞
    url = "https://tw.stock.yahoo.com/tw-market/"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    
    try:
        resp = requests.get(url, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.find_all("h3", class_="Mt(0) Mb(8px)")
        
        new_records = []
        for i in items:
            title = i.text.strip()
            if any(k.lower() in title.lower() for k in KEYWORDS):
                # AI 分析
                res = nlp_pipe(title)[0]
                label, score = res['label'], round(res['score'], 4)
                
                new_records.append({
                    "抓取時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "新聞標題": title,
                    "AI 情緒標籤": label,
                    "AI 信心度": score,
                    "當時股價": price,
                    "今日漲跌%": change
                })

        # C. 數據整合與情緒指數計算
        if new_records:
            new_df = pd.DataFrame(new_records)
            
            # 如果已有檔案則讀取合併
            if os.path.exists(CSV_FILE):
                old_df = pd.read_csv(CSV_FILE)
                df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset=["新聞標題"])
            else:
                df = new_df
            
            # 計算今日情緒指數 (Sentiment Index)
            # 權重：Positive=1, Negative=-1, Neutral=0
            weight_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
            weighted_score = df[df['抓取時間'].str.contains(datetime.now().strftime("%Y-%m-%d"))]
            index_val = round(weighted_score['AI 情緒標籤'].map(weight_map).mean() * 100, 2)
            
            # 儲存與顯示結果
            df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
            print(f"\n任務完成！")
            print(f"今日情緒指數：{index_val} (-100 ~ +100)")
            print(f"當前股價：{price} ({change}%)")
            
            # D. 更新圖表
            generate_visuals(df)
        else:
            print(" 沒發現相關新聞。")

    except Exception as e:
        print(f" 執行出錯: {e}")

if __name__ == "__main__":
    main()