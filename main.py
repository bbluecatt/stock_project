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
# 1. 系統配置 (請確認網址路徑正確)
# ==========================================
MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
TARGET_STOCK = "2330.TW"
KEYWORDS = ["台積電", "TSMC", "2330"]
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# 讀取 GitHub Secrets
LINE_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_USER_ID = os.getenv('LINE_USER_ID')

# ✨ 圖片網址加上時間戳記，解決 LINE 的快取問題 (Cache Busting)
RAW_IMG_URL = "https://raw.githubusercontent.com/bbluecatt/stock_project/main/trend.png"
IMG_URL_WITH_CACHE_BUST = f"{RAW_IMG_URL}?t={int(time.time())}"

print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 正在載入 AI 引擎...")
nlp = pipeline("sentiment-analysis", model=MODEL_NAME)

# ==========================================
# 2. 核心功能函數
# ==========================================

def get_accurate_stock_price(stock_id):
    """獲取最新股價與漲跌幅"""
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
    """生成趨勢圖，並強制存於根目錄"""
    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 5))
    
    if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
        df = pd.read_csv(file_name)
        df['抓取時間'] = pd.to_datetime(df['抓取時間'])
        df = df.sort_values('抓取時間').tail(15)
        plt.plot(df['抓取時間'], df['當時股價'], color='tab:blue', marker='o', alpha=0.4)
        plt.title(f'{TARGET_STOCK} AI Sentiment Report')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Data Initializing...', ha='center')
    
    # 強制存檔至根目錄
    plt.savefig('trend.png', bbox_inches='tight')
    plt.close()
    print("📊 圖片 trend.png 已生成")

def send_line_notification(message, image_url=None):
    """發送 LINE 訊息與圖片"""
    if not LINE_TOKEN or not LINE_USER_ID: 
        print("❌ 缺少 LINE Token 或 User ID")
        return
    line_bot_api = LineBotApi(LINE_TOKEN)
    try:
        line_bot_api.push_message(LINE_USER_ID, TextSendMessage(text=message))
        if image_url:
            line_bot_api.push_message(LINE_USER_ID, ImageSendMessage(image_url, image_url))
        print("✅ LINE 推送成功")
    except Exception as e: 
        print(f"❌ LINE 失敗: {e}")

# ==========================================
# 3. 主程式流程 (暴力掃描模式)
# ==========================================

def run_system():
    price, change = get_accurate_stock_price(TARGET_STOCK)
    url = f"https://tw.stock.yahoo.com/quote/{TARGET_STOCK}/news"
    resp = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # 搜尋所有超連結文字
    items = soup.find_all('a')
    new_data = []
    max_news = 20
    
    for item in items:
        if len(new_data) >= max_news: break
        title = item.get_text().strip()
        # 只要標題長度夠且含有關鍵字就抓
        if len(title) > 12 and any(k.lower() in title.lower() for k in KEYWORDS):
            res = nlp(title[:128])[0]
            label = "Positive" if "1" in res['label'] or "pos" in res['label'].lower() else "Negative"
            new_data.append({
                "抓取時間": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "新聞標題": title, "AI 情緒標籤": label, "當時股價": price, "漲跌": change
            })

    file_name = "stock_ai_deep_analysis.csv"
    
    # 確保 CSV 檔案存在，防止後續 Git 報錯
    if not os.path.exists(file_name):
        pd.DataFrame(columns=["抓取時間", "新聞標題", "AI 情緒標籤", "當時股價", "漲跌"]).to_csv(file_name, index=False)

    if new_data:
        df = pd.DataFrame(new_data)
        old_df = pd.read_csv(file_name)
        # 去重複
        df = pd.concat([old_df, df]).drop_duplicates(subset=["新聞標題"])
        df.to_csv(file_name, index=False, encoding="utf-8-sig")
        
        draw_and_save_chart(file_name)
        msg = f"🤖 台積電 AI 報告\n現價: {price} 元\n漲跌: {change}%\n狀態: 偵測到 {len(new_data)} 則動態！"
        send_line_notification(msg, IMG_URL_WITH_CACHE_BUST)
    else:
        # 心跳回報
        draw_and_save_chart(file_name)
        heartbeat_msg = f"🤖 系統回報\n現價: {price} 元\n漲跌: {change}%\n狀態: 目前暫無符合關鍵字的新聞。"
        send_line_notification(heartbeat_msg, None)

if __name__ == "__main__":
    run_system()