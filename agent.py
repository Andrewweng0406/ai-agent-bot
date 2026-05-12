import os
import json
import requests
import yfinance as yf
import pandas as pd
import feedparser

from flask import Flask, request
from openai import OpenAI
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
MY_LINE_USER_ID = os.getenv("MY_LINE_USER_ID")

client = OpenAI(api_key=OPENAI_KEY)
app = Flask(__name__)


# =========================
# 技術指標
# =========================

def add_indicators(df):
    df = df.copy()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20"]

    return df

def get_stock_news(symbol):
    try:
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

        feed = feedparser.parse(rss_url)

        news_list = []

        for entry in feed.entries[:5]:
            news_list.append({
                "title": entry.title,
                "summary": entry.summary if hasattr(entry, "summary") else "",
                "link": entry.link
            })

        return news_list

    except Exception as e:
        return [{"title": f"新聞抓取失敗: {str(e)}"}]

def get_market_filter():
    try:
        qqq = yf.Ticker("QQQ").history(period="6mo", interval="1d")
        spy = yf.Ticker("SPY").history(period="6mo", interval="1d")

        qqq = add_indicators(qqq).dropna()
        spy = add_indicators(spy).dropna()

        q = qqq.iloc[-1]
        s = spy.iloc[-1]

        market_score = 0
        market_status = "中性"
        market_warning = []

        if q["Close"] > q["EMA20"] > q["EMA50"]:
            market_score += 2
        else:
            market_score -= 2
            market_warning.append("QQQ 結構偏弱，科技股順風不足")

        if s["Close"] > s["EMA20"]:
            market_score += 1
        else:
            market_score -= 1
            market_warning.append("SPY 跌破 EMA20，大盤風險偏高")

        if market_score >= 2:
            market_status = "強勢"
        elif market_score <= -2:
            market_status = "偏弱"
        else:
            market_status = "震盪"

        return {
            "score": market_score,
            "status": market_status,
            "warnings": market_warning
        }

    except Exception as e:
        return {
            "score": 0,
            "status": "未知",
            "warnings": [f"大盤資料讀取失敗：{str(e)}"]
        }


def detect_swing_setup(daily, h4, h1, market):
    d = daily.iloc[-1]
    h4_last = h4.iloc[-1]
    h1_last = h1.iloc[-1]

    price = d["Close"]
    ema20 = d["EMA20"]
    ema50 = d["EMA50"]
    rsi = d["RSI"]
    atr = d["ATR"]
    vol_ratio = d["VOL_RATIO"]

    recent_high = daily["High"].iloc[-20:-1].max()
    recent_low = daily["Low"].iloc[-20:-1].min()

    score = 0
    reasons = []
    warnings = []
    setup_type = "Neutral"

    # 大盤濾網
    score += market["score"]

    if market["status"] == "強勢":
        reasons.append("大盤濾網偏強，順風交易環境較好")
    elif market["status"] == "偏弱":
        warnings.append("大盤偏弱，個股做多需要降級處理")
    else:
        warnings.append("大盤震盪，不能追高")

    # 趨勢
    if price > ema20 > ema50:
        score += 2
        reasons.append("Daily 趨勢偏多，價格站上 EMA20 / EMA50")
    elif price < ema20 < ema50:
        score -= 2
        reasons.append("Daily 趨勢偏空，價格跌破 EMA20 / EMA50")
    else:
        warnings.append("Daily 結構混亂，趨勢不夠乾淨")

    # RSI
    if 55 <= rsi <= 70:
        score += 2
        reasons.append("RSI 在健康多頭區間")
    elif rsi > 75:
        score -= 1
        warnings.append("RSI 過熱，不適合追高")
    elif rsi < 40:
        score -= 1
        warnings.append("RSI 偏弱，買方力量不足")

    # 量能
    if vol_ratio >= 1.5:
        score += 2
        reasons.append("量能明顯放大，有資金進場跡象")
    elif vol_ratio < 0.8:
        score -= 1
        warnings.append("量能不足，突破可信度偏低")

    # 突破
    if price > recent_high:
        score += 2
        reasons.append("價格突破近 20 日高點")
        setup_type = "Breakout"
    elif price < recent_low:
        score -= 2
        warnings.append("價格跌破近 20 日低點")

    # 4H setup
    if h4_last["Close"] > h4_last["EMA20"]:
        score += 1
        reasons.append("4H 仍站在 EMA20 上方")
    else:
        warnings.append("4H 尚未重新站穩 EMA20")

    # 1H 進場
    if h1_last["Close"] > h1_last["EMA20"]:
        score += 1
        reasons.append("1H 短線進場結構偏強")
    else:
        warnings.append("1H 進場點還不夠漂亮")


    entry_zone_low = ema20 - atr * 0.3
    entry_zone_high = ema20 + atr * 0.3

    stop_loss = entry_zone_low - atr * 0.8
    target_1 = entry_zone_high + atr * 2.5
    target_2 = entry_zone_high + atr * 4

    planned_entry = entry_zone_high

    risk = planned_entry - stop_loss
    reward = target_1 - planned_entry
    rr_ratio = reward / risk if risk > 0 else 0
    distance_from_entry = (price - planned_entry) / planned_entry
    
    # 硬性禁止條件
    hard_no_trade = False

    if rr_ratio < 1.5:
        hard_no_trade = True
        warnings.append("風報比太差，不值得冒這個風險")

    if rsi > 75:
        hard_no_trade = True
        warnings.append("RSI 過熱，容易追在短線高點")
    
    # Setup 類型判斷
    if abs(price - ema20) / ema20 < 0.02:
        setup_type = "Pullback"

    if distance_from_entry > 0.05:

        if (
            price > recent_high
            and vol_ratio > 1.5
            and market["score"] >= 2
        ):

            reasons.append("強勢突破結構，允許 Momentum Breakout 模式")
            setup_type = "Momentum Breakout"

        else:

            hard_no_trade = True
            warnings.append("現價距離等待區過遠，不適合追價")
            setup_type = "Overextended"

    # 評級系統
    if hard_no_trade and setup_type != "Momentum Breakout":
        rating = "🔴 No Trade"
        bias = "風險過高"
    
    elif setup_type == "Momentum Breakout":
        rating = "🟡 Momentum Watchlist"
        bias = "強勢突破"

    elif score >= 6:
        rating = "🟢 High Quality Setup"
        bias = "偏多"

    elif score >= 3:
        rating = "🟡 Watchlist / 等回踩"
        bias = "偏多但需要確認"

    elif score <= -3:
        rating = "🔴 Avoid / 偏空"
        bias = "偏空"

    else:
        rating = "🔴 No Trade"
        bias = "方向不明"

    return {
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "rsi": rsi,
        "atr": atr,
        "vol_ratio": vol_ratio,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "score": score,
        "rating": rating,
        "bias": bias,
        "reasons": reasons,
        "warnings": warnings,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "entry_zone_low": entry_zone_low,
        "entry_zone_high": entry_zone_high,
        "planned_entry": planned_entry,
        "rr_ratio": rr_ratio,
        "market_status": market["status"],
        "setup_type": setup_type,
        "distance_from_entry": distance_from_entry,
    }


# =========================
# 股票分析主函數
# =========================
def get_us_morning_briefing():
    watchlist = ["NVDA", "TSLA", "AMD", "MU", "AAPL"]
    market_symbols = ["SPY", "QQQ"]
    market = get_market_filter()

    summaries = []

    for symbol in watchlist:
        try:
            ticker = yf.Ticker(symbol)

            daily = ticker.history(period="6mo", interval="1d", auto_adjust=False)
            h4 = ticker.history(period="60d", interval="1h", auto_adjust=False)
            h1 = ticker.history(period="30d", interval="1h", auto_adjust=False)

            if daily.empty or h4.empty or h1.empty:
                continue

            daily = add_indicators(daily).dropna()
            h4 = add_indicators(h4).dropna()
            h1 = add_indicators(h1).dropna()

            setup = detect_swing_setup(daily, h4, h1, market)
            log_trade(symbol, setup)

            summaries.append(
                f"{symbol}: {setup['rating']} | {setup['setup_type']} | 現價 {setup['price']:.2f} | 等待區 {setup['entry_zone_low']:.2f}-{setup['entry_zone_high']:.2f}"
            )

        except Exception as e:
            summaries.append(f"{symbol}: 分析失敗 {str(e)}")

    news = get_stock_news("QQQ")
    news_text = "\n".join([f"- {n['title']}" for n in news[:5]])

    prompt = f"""
你是專業美股 Swing Trading 盤前簡報助理。

請根據以下資料，產生一份每天開盤前看的 Trading Briefing。

大盤狀態：
{market}

Watchlist 分析：
{summaries}

近期科技股/QQQ新聞：
{news_text}

請用繁體中文輸出，格式如下：

📌 美股開盤前 Briefing

1. 今日大盤狀態
2. 市場風險等級
3. 重要新聞摘要
4. 今日 Watchlist
5. 今日可觀察標的
6. 今日不該追的標的
7. 今日交易紀律

要求：
- 簡短清楚
- SPY 和 QQQ 只用來判斷大盤，不要列為今日可交易標的
- 等待區是回踩區，不是突破區
- 如果現價高於等待區超過 5%，請說不要追價，等待回踩
- 今日可觀察標的只能從 NVDA、TSLA、AMD、MU、AAPL 中挑
- 不要把無關 ETF 新聞當成重要市場新聞
- 不要保證獲利
- 直接告訴我今天應該偏進攻、保守、還是觀望
- 如果大盤不好，要直接提醒少做
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.35,
        max_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": "你是冷靜、專業、重視風險的美股盤前交易簡報助理。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
def log_trade(symbol, setup):

    try:

        with open("trades.json", "r") as f:
            trades = json.load(f)

    except:
        trades = []

    trade_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "setup_type": setup.get("setup_type"),
        "rating": setup.get("rating"),
        "market_status": setup.get("market_status"),
        "price": round(setup.get("price", 0), 2),
        "planned_entry": round(setup.get("planned_entry", 0), 2),
        "stop_loss": round(setup.get("stop_loss", 0), 2),
        "target_1": round(setup.get("target_1", 0), 2),
        "target_2": round(setup.get("target_2", 0), 2),
        "rr_ratio": round(setup.get("rr_ratio", 0), 2),
    }

    trades.append(trade_data)

    with open("trades.json", "w") as f:
        json.dump(trades, f, indent=4)

    print(f"📝 已紀錄 setup: {symbol}")

def get_ai_analysis(user_input):
    try:
        user_input = user_input.strip()

        if user_input.isdigit() and len(user_input) == 4:
            symbol = f"{user_input}.TW"
            is_us_stock = False
        else:
            symbol = user_input.upper()
            is_us_stock = True

        print(f"🚀 Swing 分析中: {symbol}")

        ticker = yf.Ticker(symbol)

        daily = ticker.history(period="6mo", interval="1d", auto_adjust=False)
        h4 = ticker.history(period="60d", interval="1h", auto_adjust=False)
        h1 = ticker.history(period="30d", interval="1h", auto_adjust=False)

        # 台股上櫃
        if daily.empty and not is_us_stock:
            symbol = f"{user_input}.TWO"
            ticker = yf.Ticker(symbol)
            daily = ticker.history(period="6mo", interval="1d", auto_adjust=False)
            h4 = ticker.history(period="60d", interval="1h", auto_adjust=False)
            h1 = ticker.history(period="30d", interval="1h", auto_adjust=False)

        if daily.empty or h4.empty or h1.empty:
            return f"❌ 找不到 {symbol} 的足夠資料，請檢查代號。"

        daily = add_indicators(daily).dropna()
        h4 = add_indicators(h4).dropna()
        h1 = add_indicators(h1).dropna()

        if len(daily) < 50:
            return f"❌ {symbol} 資料不足，無法做 Swing 判斷。"

        market = get_market_filter()
        setup = detect_swing_setup(daily, h4, h1, market)
        log_trade(symbol, setup)
        news = get_stock_news(symbol)

        return call_openai_swing(symbol, setup, news)

    except Exception as e:
        return f"⚠️ 系統錯誤: {str(e)}"


# =========================
# OpenAI 解說
# =========================

def call_openai_swing(symbol, setup, news):
    news_text = "\n".join(
    [f"- {n['title']}" for n in news]
)
    prompt = f"""
你是一個專業 Swing Trading 交易助理。
你不是喊單老師，你的任務是提高交易紀律，過濾爛交易。

股票：{symbol}

數據：
現價：{setup["price"]:.2f}
Daily EMA20：{setup["ema20"]:.2f}
Daily EMA50：{setup["ema50"]:.2f}
RSI：{setup["rsi"]:.2f}
ATR：{setup["atr"]:.2f}
量比：{setup["vol_ratio"]:.2f}
20日高點：{setup["recent_high"]:.2f}
20日低點：{setup["recent_low"]:.2f}
系統分數：{setup["score"]}
系統評級：{setup["rating"]}
Setup 類型：{setup["setup_type"]}
方向：{setup["bias"]}
大盤狀態：{setup["market_status"]}
風報比：{setup["rr_ratio"]:.2f}R
目前價格與計畫進場價差距：{setup["price"] - setup["planned_entry"]:.2f}

目前價格偏離等待區：
{setup["distance_from_entry"] * 100:.2f}%

近期新聞：
{news_text}

支持理由：
{setup["reasons"]}

風險警告：
{setup["warnings"]}

等待回踩區：{setup["entry_zone_low"]:.2f} - {setup["entry_zone_high"]:.2f}
參考停損：{setup["stop_loss"]:.2f}
第一目標：{setup["target_1"]:.2f}
第二目標：{setup["target_2"]:.2f}
計畫進場價：{setup["planned_entry"]:.2f}

請用繁體中文。

除了技術面，
還要分析：

1. 新聞是否真的重要
2. 對 swing trader 是短期、中期還是雜訊
3. 市場是否可能已提前反映

如果新聞沒什麼用，
請直接說「新聞面影響有限」。

如果 No Trade 的原因是「距離等待區過遠」，請不要說 Daily 結構混亂，除非 warnings 明確有這句。

如果 Setup 類型是 Momentum Breakout
請說明：
這是強勢突破單，
不是低風險回踩單。

輸出 LINE 適合閱讀的格式：

📊 股票 Swing 分析

1. 方向與目前價格
2. Setup 品質
3. 等待區與進場策略
4. 停損位置
5. 目標價與風報比
   如果系統評級是 🔴 No Trade，請不要強調目標價，只說「目前不執行目標價，因為尚未進入合理進場區」。
6. 新聞面影響
7. 不交易條件
8. 最後一句紀律提醒

要求：
- 不要保證獲利
- 不要說穩賺
- 如果 setup 不好，要直接說不要做
- 語氣專業、直接、有交易員感
- 簡短清楚

- 不要加入系統資料沒有提供的矛盾描述
- 如果系統評級是 🟢，不要說 Daily 結構混亂
- 如果 warnings 裡沒有該風險，不要自己編風險
- 一定要說明：
  1. 現價是否離等待區太遠
  2. 是否適合追價
  3. 跌破哪裡取消交易
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.35,
        max_tokens=700,
        messages=[
            {
                "role": "system",
                "content": "你是專業、冷靜、重視風險的 Swing Trading 分析助理。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# =========================
# LINE 回覆
# =========================

def reply_line(reply_token, text):
    url = "https://api.line.me/v2/bot/message/reply"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }

    payload = {
        "replyToken": reply_token,
        "messages": [
            {
                "type": "text",
                "text": text[:4900]
            }
        ]
    }

    r = requests.post(url, headers=headers, json=payload)
    print("LINE reply status:", r.status_code, r.text)

def push_line(user_id, text):
    url = "https://api.line.me/v2/bot/message/push"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }

    payload = {
        "to": user_id,
        "messages": [
            {
                "type": "text",
                "text": text[:4900]
            }
        ]
    }

    r = requests.post(url, headers=headers, json=payload)
    print("LINE push status:", r.status_code, r.text)


# =========================
# Flask routes
# =========================

@app.route("/", methods=["GET"])
def home():
    return "SWING BOT IS RUNNING", 200

@app.route("/callback", methods=["GET", "POST"])
def callback():
    if request.method == "GET":
        return "OK", 200

    body = request.get_data(as_text=True)
    print("BODY:", body)

    try:
        data = json.loads(body)

        for event in data.get("events", []):
            if event.get("type") != "message":
                continue

            message = event.get("message", {})

            if message.get("type") != "text":
                continue

            user_msg = message.get("text", "").strip()
            reply_token = event.get("replyToken")

            print(f"💬 查詢: {user_msg}")

            if user_msg.lower() in ["/morning", "morning"]:
                briefing = get_us_morning_briefing()
                reply_line(reply_token, briefing)

            elif user_msg.lower() in ["/help", "help", "幫助"]:
                help_text = """
📌 Swing Bot 使用方式

直接輸入股票代號：

美股：
TSLA
NVDA
AAPL
AMD
MU

台股：
2330
2317

/morning
查看今日盤前簡報
"""
                reply_line(reply_token, help_text)

            else:
                analysis = get_ai_analysis(user_msg)
                reply_line(
                    reply_token,
                    f"📊 {user_msg.upper()} Swing 分析報告：\n\n{analysis}"
                )

    except Exception as e:
        print("❌ Callback error:", e)

    return "OK", 200


def send_morning_briefing():
    try:
        if not MY_LINE_USER_ID:
            print("❌ MY_LINE_USER_ID 沒有設定")
            return

        briefing = get_us_morning_briefing()
        push_line(MY_LINE_USER_ID, briefing)

    except Exception as e:
        print("❌ Morning briefing error:", e)

if __name__ == "__main__":
    scheduler = BackgroundScheduler(timezone="America/Los_Angeles")

    # 美股開盤前 30 分鐘，加州時間 6:00 AM
    scheduler.add_job(
        send_morning_briefing,
        "cron",
        day_of_week="mon-fri",
        hour=6,
        minute=0
    )

    scheduler.start()

    app.run(host="0.0.0.0", port=5001)