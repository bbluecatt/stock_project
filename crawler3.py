import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import time
import os
from datetime import datetime
from transformers import pipeline

# ==========================================
# 1. 系統配置
# ==========================================
MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
TARGET_STOCK = "2330.TW"  # 監測台積電
KEYWORDS = ["台積電", "TSMC", "2330", "AI", "半導體", "輝達", "NVIDIA", "晶片", "先進製程"]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 正在初始化 AI 大腦 (二郎神模型)...")
nlp = pipeline("sentiment-analysis", model=MODEL_NAME)

# ==========================================
# 2. 功能函數定義
# ==========================================

def get_accurate_stock_price(stock_id):
    """獲取精確股價，具備多重備援機制確保數據不消失"""
    try:
        stock = yf.Ticker(stock_id)
        # 優先嘗試 1：從歷史數據計算漲跌 (最穩定)
        df = stock.history(period="2d")
        if len(df) >= 2:
            prev_close = df['Close'].iloc[-2]
            current_price = df['Close'].iloc[-1]
            change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)
            return round(current_price, 2), change_pct
        
        # 備援嘗試 2：如果歷史數據不足，直接抓取即時資訊
        else:
            current_price = stock.fast_info.get('last_price')
            prev_close = stock.info.get('previousClose')
            if current_price and prev_close:
                change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)
                return round(current_price, 2), change_pct
    except Exception as e:
        print(f"   ⚠️ 股價抓取異常: {e}")
    
    return "N/A", "N/A"

def get_article_content(url):
    """多重容器偵測：確保能點進網址並抓到新聞內文"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=7)
        if r.status_code != 200: return ""
        soup = BeautifulSoup(r.text, "html.parser")
        
        possible_selectors = [
            ("div", "caas-body"),      # Yahoo 標準
            ("div", "article-body"),   # 外部媒體
            ("article", ""),           # HTML5 標準
            ("div", "canvas-body")     # 影音圖文版
        ]
        
        for tag, class_name in possible_selectors:
            content_div = soup.find(tag, class_=class_name) if class_name else soup.find(tag)
            if content_div:
                text = content_div.get_text(separator=' ', strip=True)
                if len(text) > 50:
                    return text[:500] 
    except:
        pass
    return ""

# ==========================================
# 3. 主流程邏輯
# ==========================================

def run_system():
    print(f"\n{'='*40}")
    print(f"📅 任務執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # STEP 1: 抓取行情 (在此確保價格不會消失)
    price, change = get_accurate_stock_price(TARGET_STOCK)
    print(f"📈 {TARGET_STOCK} 當前行情: {price} 元 (漲跌: {change}%)")

    # STEP 2: 抓取新聞清單
    list_url = "https://tw.stock.yahoo.com/tw-market/"
    try:
        resp = requests.get(list_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        news_items = soup.find_all("div", class_="Py(14px)")
    except Exception as e:
        print(f"❌ 無法讀取新聞清單: {e}")
        return

    new_data = []
    processed_count = 0
    
    for item in news_items:
        if processed_count >= 8: break 
        
        link_tag = item.find("a", href=True)
        if link_tag and "/news/" in link_tag['href']:
            title = link_tag.text.strip()
            
            # --- 關鍵字過濾 ---
            if not any(k.lower() in title.lower() for k in KEYWORDS):
                continue 
            
            href = link_tag['href']
            full_url = href if href.startswith('http') else "https://tw.stock.yahoo.com" + href
            
            print(f"\n🎯 發現相關新聞: {title[:25]}...")
            
            # STEP 3: 進入內文抓取
            print("   📡 讀取內文中...", end="", flush=True)
            content = get_article_content(full_url)
            
            if content:
                print(" [OK]")
                # STEP 4: AI 深度分析
                print(f"   🧠 AI 深度分析中...", end="", flush=True)
                analysis = nlp(content)[0]
                
                label = "Positive" if "1" in analysis['label'] or "pos" in analysis['label'].lower() else "Negative"
                score = round(analysis['score'], 4)
                print(f" [判定: {label} / 信心度: {score}]")
                
                new_data.append({
                    "抓取時間": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "新聞標題": title,
                    "AI 情緒標籤": label,
                    "AI 信心度": score,
                    "當時股價": price,    # 這裡確保存入剛抓到的價錢
                    "今日漲跌%": change,  # 這裡確保存入剛抓到的漲跌
                    "內文摘要": content[:50].replace("\n", " ") + "..."
                })
                processed_count += 1
                time.sleep(2.5) 
            else:
                print(" [跳過: 無法解析內文]")

    # STEP 6: 資料存檔與去重
    if new_data:
        df = pd.DataFrame(new_data)
        file_name = "stock_ai_deep_analysis.csv"
        
        if os.path.exists(file_name):
            old_df = pd.read_csv(file_name)
            df = pd.concat([old_df, df], ignore_index=True).drop_duplicates(subset=["新聞標題"])
        
        df.to_csv(file_name, index=False, encoding="utf-8-sig")
        print(f"\n✅ 任務完成！最新資料已同步至 {file_name}")
    else:
        print("\n💡 本次掃描無符合關鍵字的新數據。")

if __name__ == "__main__":
    run_system()