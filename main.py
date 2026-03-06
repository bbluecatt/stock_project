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
KEYWORDS = ["台積電", "TSMC", "2330", "AI", "半導體", "晶片", "營收", "法說"]
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

LINE_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_USER_ID = os.getenv('LINE_USER_ID')
IMG_URL = "https://raw.githubusercontent.com/bbluecatt/stock_project/main/trend.png"

print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 正在初始化 AI 大腦...")
nlp = pipeline("sentiment-analysis", model=MODEL_NAME)

# ==========================================
# 2. 功能函數
# ==========================================

def get_accurate_stock_price(stock_id):
    """確保漲跌幅與價格都能抓到"""
    try:
        stock = yf.Ticker(stock_id)
        df = stock.history(period="2d")
        if len(df) >= 2:
            prev_close = df['Close'].iloc[-2]
            current_price = df['Close'].iloc[-1]
            change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)
            return round(current_price, 2), change_pct
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
    plt.title(f'TSMC (2330) Trend Analysis')
    plt.savefig('trend.png')

def send_line_notification(message, image_url=None):
    if not LINE_TOKEN or not LINE_USER_ID: return
    line_bot_api = LineBotApi(LINE_TOKEN)
    try:
        line_bot_api.push_message(LINE_USER_ID, TextSendMessage(text=message))
        if image_url:
            line_bot_api.push_message(LINE_USER_ID, ImageSendMessage(image_url, image_url))
        print("✅ LINE 訊息已成功推播！")
    except Exception as e: print(f"❌ LINE 發送失敗: {e}")

# ==========================================
# 3. 主程式邏輯
# ==========================================

def run_system():
    price, change = get_accurate_stock_price(TARGET_STOCK)
    
    # 掃描多個可能的新聞區塊
    urls = [f"https://tw.stock.yahoo.com/quote/{TARGET_STOCK}/news", "https://tw.stock.yahoo.com/news/"]
    new_data = []
    
    for url in urls:
        resp = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(resp.text, "html.parser")
        # 廣泛搜尋標題區塊
        items = soup.find_all(['h3', 'a'], class_=lambda x: x and any(c in x for c in ['Title', 'Py(14px)', 'List(n)']))
        
        for item in items:
            title = item.get_text().strip()
            if len(title) > 10 and any(k in title for k in KEYWORDS):
                # AI 分析
                res = nlp(title[:128])[0]
                label = "Positive" if "1" in res['label'] or "pos" in res['label'].lower() else "Negative"
                new_data.append({
                    "抓取時間": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "新聞標題": title, "AI 情緒標籤": label, "當時股價": price, "漲跌": change
                })
        if new_data: break

    file_name = "stock_ai_deep_analysis.csv"
    if new_data:
        df = pd.DataFrame(new_data)
        if os.path.exists(file_name):
            old_df = pd.read_csv(file_name)
            df = pd.concat([old_df, df]).drop_duplicates(subset=["新聞標題"])
        df.to_csv(file_name, index=False, encoding="utf-8-sig")
        
        draw_and_save_chart(file_name)
        # ✨ 確保漲跌幅出現在訊息中
        msg = f"🤖 台積電 AI 報告\n現價: {price} 元\n漲跌: {change}%\n狀態: 偵測到 {len(new_data)} 則動態！"
        send_line_notification(msg, IMG_URL)
    else:
        # 心跳回報也包含漲跌幅
        heartbeat_msg = f"🤖 系統回報\n現價: {price} 元\n漲跌: {change}%\n狀態: 目前暫無新相關新聞。"
        send_line_notification(heartbeat_msg, None)

if __name__ == "__main__":
    run_system()