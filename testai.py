from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 1. 載入二郎神模型
MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
print("正在載入 AI 大腦...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def test_sentiment(text):
    # AI 模型通常有 512 個字的限制，我們取前 500 字
    clean_text = text.replace("\n", "").strip()[:500]
    result = nlp(clean_text)[0]
    
    label = result['label']
    score = round(result['score'], 4)
    
    # 轉換標籤名稱
    display_label = "正面 (Positive)" if "1" in label or "positive" in label.lower() else "負面 (Negative)"
    
    print("-" * 30)
    print(f"分析結果：{display_label}")
    print(f"AI 信心度：{score}")
    print("-" * 30)

# ==========================================
# 2. 在這裡貼上你想測試的「新聞標題」或「內文」
# ==========================================
print("\n--- 測試 1：只測標題 ---")
test_sentiment("台積電飆破2000元！有人唱衰「會倒不敢買」　不敗教主：台灣人還在信這種鬼話")

print("\n--- 測試 2：測試內文 (請把網頁上的內文貼在下面) ---")
news_content = """
台股從開紅盤後持續突破天際，今（25）日指數一口氣大漲712點，以3萬5413點新天價作收，權值股台積電更是飆破2000元大關，收盤站上2015元。對此，投資專家「不敗教主」陳重銘直言，有些人還覺得「台積電會倒」，台積電有7成股權掌握在外國人手上，「就是因為你還在相信台積電會倒這種鬼話！」

陳重銘表示，沒想到還可以看到台積電上2000，全球AI大爆發，所有的科技產品都需要IC，沒有專業晶圓代工的台積電，不管是蘋果還是輝達，都賺不到錢！台積電很聰明的不發展自己品牌，選擇代工這條路，讓所有的大廠都離不開它！
受惠於輝達訂單與全球 AI 需求，基本面依然強勁，長線看好...
陳重銘指出，輝達、高通為何不成立自己的晶圓廠？他們有錢但是沒有這個技術，連專業的三星跟英特爾，都只能看台積電的車尾燈，所以這些世界大廠，只能乖乖跟台積電合作，把訂單源源不絕送給台積電。
陳重銘直言，有些人覺得台積電會倒，還是買ETF比較安全，「你真的瞭解台積電嗎？」全世界的科技大廠，蘋果、輝達、高通、特斯拉等，都沒有自己的晶圓廠，所有的高階晶片，都要靠台積電提供。
陳重銘提到，為什麼台積電要到日本、德國、美國設廠？因為這些國家要確保台積電給他晶片，「沒有台積電，科技業會完蛋，這些國家清楚得很，特別是美國。」
陳重銘強調，台灣難得孵育出全球第6大企業，但是7成台積電股權掌握在外國人手上，「台灣人你為何不買台積電？因為你還在相信台積電會倒這種鬼話！7成的外資怎麼都不怕？」
"""
test_sentiment(news_content)