import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
from transformers import pipeline

# 1. 初始化設定
# 載入本地 AI 模型 (第一次執行會下載約 400MB 模型)
print("正在載入本地 FinBERT 財經 AI 模型...")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# 設定目標與關鍵字
TARGET_STOCK = "2330.TW"  # 台積電
KEYWORDS = ["台積電", "TSMC", "AI", "半導體", "輝達", "營收", "噴出"]
URL = "https://tw.stock.yahoo.com/tw-market/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_live_stock_info(stock_id):
    """抓取目前的即時股價與漲跌幅"""
    try:
        stock = yf.Ticker(stock_id)
        df_price = stock.history(period="1d")
        if not df_price.empty:
            close_p = round(df_price['Close'].iloc[-1], 2)
            # 計算今日漲跌百分比 (相較於開盤)
            open_p = df_price['Open'].iloc[-1]
            change_pct = round(((close_p - open_p) / open_p) * 100, 2)
            return close_p, change_pct
    except Exception as e:
        print(f"股價抓取失敗: {e}")
    return "N/A", "N/A"

def start_ai_stock_system():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 啟動 AI 智能監測系統...")
    
    # A. 抓取即時股價
    current_price, daily_change = get_live_stock_info(TARGET_STOCK)
    
    # B. 執行爬蟲抓取新聞
    try:
        response = requests.get(URL, headers=HEADERS)
        if response.status_code != 200:
            print(f" 無法連線至網站: {response.status_code}")
            return

        soup = BeautifulSoup(response.text, "html.parser")
        news_items = soup.find_all("h3", class_="Mt(0) Mb(8px)")
        
        new_data = []
        for item in news_items:
            title = item.text.strip()
            
            # C. 關鍵字篩選
            if any(word.lower() in title.lower() for word in KEYWORDS):
                # D. 本地 AI 情感分析
                ai_result = finbert(title)[0]
                label = ai_result['label']
                confidence = round(ai_result['score'], 3)
                
                print(f" 發現相關新聞: {title[:20]}... -> AI 判斷: {label}")
                
                new_data.append({
                    "抓取時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "新聞標題": title,
                    "AI 情緒標籤": label,
                    "AI 信心度": confidence,
                    "當時股價": current_price,
                    "今日漲跌%": daily_change
                })

        # E. 資料去重與存檔
        if new_data:
            df_new = pd.DataFrame(new_data)
            file_path = "stock_ai_database.csv"
            
            if os.path.isfile(file_path):
                df_old = pd.read_csv(file_path)
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                # 根據新聞標題去重，確保資料庫唯一性
                df_final = df_combined.drop_duplicates(subset=["新聞標題"], keep="first")
                added = len(df_final) - len(df_old)
            else:
                df_final = df_new
                added = len(df_final)

            df_final.to_csv(file_path, index=False, encoding="utf-8-sig")
            print(f" 處理完成！新增了 {added} 則新聞。目前總資料量：{len(df_final)}")
        else:
            print("💡 目前沒有符合關鍵字的新聞。")

    except Exception as e:
        print(f" 系統運行錯誤: {e}")

if __name__ == "__main__":
    start_ai_stock_system()