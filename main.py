import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import time
import os
from datetime import datetime
from transformers import pipeline
import matplotlib.pyplot as plt
from linebot import LineBotApi
from linebot.models import TextSendMessage, ImageSendMessage

# ==========================================
# 1. 系統配置
# ==========================================
MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
TARGET_STOCK = "2330.TW"
# 縮減關鍵字，只要有「台積電」就抓，確保一定有數據
KEYWORDS = ["台積電", "TSMC", "2330", "半導體"] 
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

LINE_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_USER_ID = os.getenv('LINE_USER_ID')
IMG_URL = "https://raw.githubusercontent.com/bbluecatt/stock_project/main/trend.png"

print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 正在啟動 AI 引擎...")
nlp = pipeline("sentiment-analysis", model=MODEL_NAME)

# ==========================================
# 2. 功能函數
# ==========================================

def get_accurate_stock_price(stock_id):
    try:
        stock = yf.Ticker(stock_id)
        df = stock.history(period="2d")
        if len(df) >= 2:
            return round(df['Close'].iloc[-1], 2), round(((df['Close'].iloc[-1]-df['Close'].iloc[-2])/df['Close'].iloc[-2])*100, 2)
    except: pass
    return "N/A", "N/A"

def draw_and_save_chart(file_name):
    if not os.path.exists(file_name): return
    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    df = pd.read_csv(file_name)
    df['抓取時間'] = pd.to_datetime(df['抓取時間'])
    df = df.sort_values('抓取時間').tail(15)
    plt.figure(figsize=(10, 5))
    plt.plot(df['抓取時間'], df['當時股價'], color='tab:blue', marker='o', alpha=0.4)
    plt.title(f'TSMC AI Analysis Success')
    plt.savefig('trend.png')

def send_line_notification(message, image_url=None):
    if not LINE_TOKEN or not LINE_USER_ID: return
    line_bot_api = LineBotApi(LINE_TOKEN)
    try:
        line_bot_api.push_message(LINE_USER_ID, TextSendMessage(text=message))
        if image_url:
            line_bot_api.push_message(LINE_USER_ID, ImageSendMessage(image_url, image_url))
        print("✅ LINE 推送成功")
    except Exception as e: print(f"❌ LINE 失敗: {e}")

# ==========================================
# 3. 強力爬蟲邏輯
# ==========================================

def run_system():
    price, change = get_accurate_stock_price(TARGET_STOCK)
    # 嘗試三個不同的新聞入口網址
    urls = [
        f"https://tw.stock.yahoo.com/quote/{TARGET_STOCK}/news",
        "https://tw.stock.yahoo.com/news/"
    ]
    
    new_data = []
    
    for url in urls:
        print(f"正在掃描: {url}")
        resp = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # 暴力搜索所有可能的標題標籤 (h3, a, span)
        potential_titles = soup.find_all(['h3', 'a'], class_=lambda x: x and ('Title' in x or 'Py(14px)' in x))
        
        for item in potential_titles:
            title_text = item.get_text().strip()
            # 只要標題長度大於 10 且包含台積電就抓
            if len(title_text) > 10 and any(k in title_text for k in KEYWORDS):
                res = nlp(title_text[:100])[0] # 直接對標題做分析提升速度
                label = "Positive" if "1" in res['label'] or "pos" in res['label'].lower() else "Negative"
                new_data.append({
                    "抓取時間": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "新聞標題": title_text, "AI 情緒標籤": label, "當時股價": price
                })
        if new_data: break # 只要抓到一個來源就停止

    file_name = "stock_ai_deep_analysis.csv"
    
    # ✨ 核心除錯動作：不論有無新數據，這次都「強制畫圖發送」
    if new_data:
        df = pd.DataFrame(new_data)
        if os.path.exists(file_name):
            old_df = pd.read_csv(file_name)
            df = pd.concat([old_df, df]).drop_duplicates(subset=["新聞標題"])
        df.to_csv(file_name, index=False, encoding="utf-8-sig")
        draw_and_save_chart(file_name)
        msg = f"🤖 AI 報表更新\n現價: {price}\n抓到 {len(new_data)} 則動態！"
        send_line_notification(msg, IMG_URL)
    else:
        # 如果還是抓不到，回報詳細狀態
        send_line_notification(f"🤖 偵錯回報\n股價: {price}\n狀態: 標籤掃描完成但未命中關鍵字。請檢查 Yahoo 頁面結構。", None)

if __name__ == "__main__":
    run_system()