import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import time
import os
from datetime import datetime
from transformers import pipeline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from linebot import LineBotApi
from linebot.models import TextSendMessage, ImageSendMessage

# ==========================================
# 1. 系統配置與環境變數
# ==========================================
MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
TARGET_STOCK = "2330.TW"
# 關鍵字維持，但因為網址改了，命中率會變極高
KEYWORDS = ["台積電", "TSMC", "2330", "AI", "半導體", "輝達", "NVIDIA", "晶片", "營收", "法說"]
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

LINE_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_USER_ID = os.getenv('LINE_USER_ID')
IMG_URL = "https://raw.githubusercontent.com/bbluecatt/stock_project/main/trend.png"

print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 正在載入 AI 大腦...")
nlp = pipeline("sentiment-analysis", model=MODEL_NAME)

# ==========================================
# 2. 功能函數
# ==========================================

def get_accurate_stock_price(stock_id):
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

def get_article_content(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=7)
        soup = BeautifulSoup(r.text, "html.parser")
        # 優化內文抓取邏輯
        content_div = soup.find("div", class_="caas-body") or soup.find("article")
        return content_div.get_text(strip=True)[:500] if content_div else ""
    except: return ""

def draw_and_save_chart(file_name):
    if not os.path.exists(file_name): return
    plt.rcParams['font.sans-serif'] = ['Heiti TC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    df = pd.read_csv(file_name)
    df['抓取時間'] = pd.to_datetime(df['抓取時間'])
    df = df.sort_values('抓取時間').tail(15)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df['抓取時間'], df['當時股價'], color='tab:blue', marker='o', alpha=0.4)
    pos = df[df['AI 情緒標籤'] == 'Positive']
    neg = df[df['AI 情緒標籤'] == 'Negative']
    ax1.scatter(pos['抓取時間'], pos['當時股價'], color='red', s=100, label='AI 正面')
    ax1.scatter(neg['抓取時間'], neg['當時股價'], color='green', s=100, label='AI 負面')
    plt.title(f'TSMC (2330) AI Sentiment Report')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('trend.png')

def send_line_notification(message, image_url=None):
    if not LINE_TOKEN or not LINE_USER_ID: return
    line_bot_api = LineBotApi(LINE_TOKEN)
    try:
        line_bot_api.push_message(LINE_USER_ID, TextSendMessage(text=message))
        if image_url:
            line_bot_api.push_message(LINE_USER_ID, ImageSendMessage(image_url, image_url))
    except Exception as e: print(f"❌ LINE 失敗: {e}")

# ==========================================
# 3. 主流程邏輯 (更新目標網址)
# ==========================================

def run_system():
    price, change = get_accurate_stock_price(TARGET_STOCK)
    # ✨ 改成台積電個股新聞分頁，抓取更精準
    list_url = f"https://tw.stock.yahoo.com/quote/{TARGET_STOCK}/news"
    resp = requests.get(list_url, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # 優化標籤選擇器以符合個股新聞頁面結構
    news_items = soup.find_all("li", class_="List(n)") 
    if not news_items:
        news_items = soup.find_all("div", class_="Py(14px)")

    new_data = []
    processed_count = 0
    
    for item in news_items:
        if processed_count >= 5: break 
        link_tag = item.find("a", href=True)
        if link_tag and "/news/" in link_tag['href']:
            title = link_tag.get_text().strip()
            # 關鍵字檢查
            if not any(k.lower() in title.lower() for k in KEYWORDS): continue
            
            full_url = link_tag['href']
            if not full_url.startswith('http'):
                full_url = "https://tw.stock.yahoo.com" + full_url
                
            content = get_article_content(full_url)
            if content:
                res = nlp(content)[0]
                label = "Positive" if "1" in res['label'] or "pos" in res['label'].lower() else "Negative"
                new_data.append({
                    "抓取時間": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "新聞標題": title, "AI 情緒標籤": label, "當時股價": price
                })
                processed_count += 1
                time.sleep(1)

    file_name = "stock_ai_deep_analysis.csv"
    if new_data:
        df = pd.DataFrame(new_data)
        if os.path.exists(file_name):
            old_df = pd.read_csv(file_name)
            df = pd.concat([old_df, df]).drop_duplicates(subset=["新聞標題"])
        df.to_csv(file_name, index=False, encoding="utf-8-sig")
        
        draw_and_save_chart(file_name)
        msg = f"🤖 台積電 AI 分析報告\n現價: {price} 元\n漲跌: {change}%\n狀態: 發現 {processed_count} 則新動態！"
        send_line_notification(msg, IMG_URL)
    else:
        # 心跳機制：即使沒新新聞也回報
        heartbeat_msg = f"🤖 系統定時回報\n現價: {price} 元\n漲跌: {change}%\n狀態: 台積電專區暫無新符合關鍵字的新聞。"
        send_line_notification(heartbeat_msg, None)

if __name__ == "__main__":
    run_system()