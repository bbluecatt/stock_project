import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
from transformers import pipeline
from linebot import LineBotApi
from linebot.models import TextSendMessage

# ==========================================
# 1. 系統配置
# ==========================================
MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
TARGET_STOCK = "2330.TW"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

LINE_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_USER_ID = os.getenv('LINE_USER_ID')

print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 正在載入深度分析 AI...")
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

def fetch_article_content(url):
    """進入網址抓取新聞內文"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all('p')
        content = "".join([p.get_text() for p in paragraphs])
        return content[:500] 
    except: return ""

def send_line_notification(message):
    if not LINE_TOKEN or not LINE_USER_ID: return
    line_bot_api = LineBotApi(LINE_TOKEN)
    try:
        line_bot_api.push_message(LINE_USER_ID, TextSendMessage(text=message))
        print("✅ LINE 推送成功")
    except Exception as e: print(f"❌ LINE 失敗: {e}")

# ==========================================
# 3. 核心邏輯 (深度閱讀模式)
# ==========================================

def run_system():
    price, change = get_accurate_stock_price(TARGET_STOCK)
    list_url = f"https://tw.stock.yahoo.com/quote/{TARGET_STOCK}/news"
    resp = requests.get(list_url, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")
    
    news_items = soup.find_all('a', href=True)
    analysis_results = []
    seen_links = set()
    processed_count = 0
    
    for item in news_items:
        if processed_count >= 5: break
        title = item.get_text().strip()
        link = item.get('href')
        
        if len(title) > 15 and "/news/" in link and link not in seen_links:
            if not link.startswith('http'):
                link = "https://tw.stock.yahoo.com" + link
            
            print(f"📖 正在深度閱讀: {title[:15]}...")
            content = fetch_article_content(link)
            
            if len(content) > 50:
                res = nlp(content[:128])[0]
                is_pos = "1" in res['label'] or "pos" in res['label'].lower()
                sentiment = "📈 看多" if is_pos else "📉 看空"
                analysis_results.append(f"● {title[:20]}...\n  AI 判斷: {sentiment}")
                seen_links.add(link)
                processed_count += 1

    report = f"🤖 台積電深度 AI 分析報告\n💰 目前股價: {price} 元 ({change}%)\n----------------------\n"
    if analysis_results:
        report += "\n".join(analysis_results)
        pos_count = sum(1 for r in analysis_results if "看多" in r)
        advice = "🚀 綜合建議: 強力看多" if pos_count >= 3 else "⚖️ 綜合建議: 謹慎保守"
        report += f"\n\n{advice}"
    else:
        report += "⚠️ 暫未發現足夠深度的相關新聞。"

    send_line_notification(report)

if __name__ == "__main__":
    run_system()