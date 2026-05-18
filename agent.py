import os
import re
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
from market_router import MarketRouter, Market

user_memory = {}
_router = MarketRouter()

# Load .env from script directory
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
LINE_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
MY_LINE_USER_ID = os.getenv("MY_LINE_USER_ID")

# Debug: Log loaded tokens
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
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20"]

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    df["BB_Mid"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
    df["BB_Pos"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    df["EMA5"]   = df["Close"].ewm(span=5,   adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    return df

def get_earnings_warning(symbol: str) -> str | None:
    """
    如果財報日在 7 天內，回傳警告字串；否則回傳 None。
    用於即時查詢（/watch 或一般分析）。
    """
    try:
        ticker = yf.Ticker(symbol)

        # 嘗試 earnings_dates（新版 yfinance）
        try:
            ed = ticker.earnings_dates
            if ed is not None and not ed.empty:
                now    = pd.Timestamp.now(tz="UTC")
                future = ed[ed.index > now]
                if not future.empty:
                    # earnings_dates 是降序排列，最近未來的在 tail
                    next_dt = future.index[-1]
                    days    = int((next_dt.tz_convert("UTC") - now).days)
                    if 0 <= days <= 7:
                        return (
                            f"⚠️ 財報警告：{symbol} 將於 {days} 天後"
                            f"（{next_dt.strftime('%m/%d')}）公布財報，\n"
                            "財報前後波動劇烈，請注意部位風險！"
                        )
        except Exception:
            pass

        # Fallback：calendar dict
        cal = ticker.calendar
        if isinstance(cal, dict) and "Earnings Date" in cal:
            dates = cal["Earnings Date"]
            if dates:
                next_dt = pd.Timestamp(dates[0])
                days    = int((next_dt - pd.Timestamp.now()).days)
                if 0 <= days <= 7:
                    return (
                        f"⚠️ 財報警告：{symbol} 將於 {days} 天後"
                        f"（{next_dt.strftime('%m/%d')}）公布財報，\n"
                        "財報前後波動劇烈，請注意部位風險！"
                    )
    except Exception:
        pass
    return None


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
        # 用 1y 確保有足夠資料計算 EMA200（大盤濾網鐵律二）
        qqq = yf.Ticker("QQQ").history(period="1y", interval="1d")
        spy = yf.Ticker("SPY").history(period="1y", interval="1d")

        qqq = add_indicators(qqq).dropna()
        spy = add_indicators(spy).dropna()

        q = qqq.iloc[-1]
        s = spy.iloc[-1]

        market_score = 0
        market_status = "中性"
        market_warning = []
        storm_mode = False
        storm_reason = ""

        # ── 鐵律二：大盤 MA200 風暴防禦 ────────────────────────
        spy_below_ma200 = s["Close"] < s.get("EMA200", s["Close"])
        qqq_below_ma200 = q["Close"] < q.get("EMA200", q["Close"])

        if spy_below_ma200 and qqq_below_ma200:
            storm_mode = True
            storm_reason = (
                "🌪️ SPY 和 QQQ 同時跌破 200 日均線，市場正在刮超強颱風！"
                "現在市場正在刮颱風，外面很危險，AI 建議您現在抱著現金好好休息，"
                "等大晴天我們再出來開工！"
            )
            market_score -= 4
            market_warning.append("⛔ SPY + QQQ 同破 MA200，進入熊市結構，嚴禁做多")
        elif spy_below_ma200:
            storm_mode = True
            storm_reason = (
                "⛈️ SPY 跌破 200 日均線，大盤進入颱風警報！"
                "現在市場正在刮颱風，外面很危險，AI 建議您現在抱著現金好好休息，"
                "等大晴天我們再出來開工！"
            )
            market_score -= 3
            market_warning.append("⛔ SPY 跌破 MA200，大盤熊市結構確立")
        elif qqq_below_ma200:
            storm_mode = True
            storm_reason = (
                "⛈️ QQQ 跌破 200 日均線，科技股進入颱風警報！"
                "現在市場正在刮颱風，外面很危險，AI 建議您減少曝險，以現金防禦為主。"
            )
            market_score -= 2
            market_warning.append("⛔ QQQ 跌破 MA200，科技股熊市結構")

        # ── 短中期大盤評估（原有邏輯）──────────────────────────
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
            "score":        market_score,
            "status":       market_status,
            "warnings":     market_warning,
            "storm_mode":   storm_mode,
            "storm_reason": storm_reason,
        }

    except Exception as e:
        return {
            "score":        0,
            "status":       "未知",
            "warnings":     [f"大盤資料讀取失敗：{str(e)}"],
            "storm_mode":   False,
            "storm_reason": "",
        }


def detect_swing_setup(daily, h4, h1, market):
    d        = daily.iloc[-1]
    h4_last  = h4.iloc[-1]
    h1_last  = h1.iloc[-1]

    price       = d["Close"]
    open_price  = d.get("Open",  price)
    high_price  = d.get("High",  price)
    low_price   = d.get("Low",   price)
    ema20       = d["EMA20"]
    ema50       = d["EMA50"]
    ema5        = d.get("EMA5", price)
    rsi         = d["RSI"]
    atr         = d["ATR"]
    vol_ratio   = d["VOL_RATIO"]

    recent_high = daily["High"].iloc[-20:-1].max()
    recent_low  = daily["Low"].iloc[-20:-1].min()

    # ── 初始化 ─────────────────────────────────────────────
    hard_no_trade    = False
    overheat_alert   = False
    distribution_alert = False
    storm_mode       = market.get("storm_mode", False)
    score            = 0
    reasons          = []
    warnings         = []
    setup_type       = "Neutral"

    # ══════════════════════════════════════════════════════
    # 鐵律一：追高強行熔斷（正乖離 > 15%）
    # ══════════════════════════════════════════════════════
    deviation_from_ma20_pct = (price - ema20) / ema20 * 100 if ema20 > 0 else 0

    if deviation_from_ma20_pct > 15:
        hard_no_trade  = True
        overheat_alert = True
        warnings.append(
            f"🔥【追高熔斷】正乖離率 {deviation_from_ma20_pct:.1f}%！"
            "現在這檔股票太熱太貴了，我們去買會幫別人洗碗，"
            "請耐心等它降溫拉回再考慮！"
        )

    # ══════════════════════════════════════════════════════
    # 鐵律二：大盤風暴防禦（MA200 跌破）
    # ══════════════════════════════════════════════════════
    if storm_mode:
        hard_no_trade = True
        warnings.append(market.get(
            "storm_reason",
            "大盤跌破 200 日均線，啟動防禦模式，請抱現金休息！"
        ))

    # ── 大盤濾網評分（原有邏輯）────────────────────────────
    score += market["score"]
    if market["status"] == "強勢":
        reasons.append("大盤濾網偏強，順風交易環境較好")
    elif market["status"] == "偏弱":
        warnings.append("大盤偏弱，個股做多需要降級處理")
    else:
        warnings.append("大盤震盪，不能追高")

    # ── 趨勢 ────────────────────────────────────────────
    if price > ema20 > ema50:
        score += 2
        reasons.append("Daily 趨勢偏多，價格站上 EMA20 / EMA50")
    elif price < ema20 < ema50:
        score -= 2
        warnings.append("Daily 趨勢偏空，價格跌破 EMA20 / EMA50")
    else:
        warnings.append("Daily 結構混亂，趨勢不夠乾淨")

    # ── RSI ─────────────────────────────────────────────
    if 55 <= rsi <= 70:
        score += 2
        reasons.append("RSI 在健康多頭區間")
    elif rsi > 75:
        score -= 1
        warnings.append("RSI 過熱，不適合追高")
    elif rsi < 40:
        score -= 1
        warnings.append("RSI 偏弱，買方力量不足")

    # ── 量能 ─────────────────────────────────────────────
    if vol_ratio >= 1.5:
        score += 2
        reasons.append("量能明顯放大，有資金進場跡象")
    elif vol_ratio < 0.8:
        score -= 1
        warnings.append("量能不足，突破可信度偏低")

    # ── MACD ─────────────────────────────────────────────
    macd_hist   = d.get("MACD_Hist",   0)
    macd_val    = d.get("MACD",        0)
    macd_signal = d.get("MACD_Signal", 0)
    if macd_hist > 0 and macd_val > macd_signal:
        score += 1
        reasons.append("MACD 柱狀圖翻正，動能轉強")
    elif macd_hist < 0:
        score -= 1
        warnings.append("MACD 動能仍偏弱")

    # ── Bollinger Bands ──────────────────────────────────
    bb_pos = d.get("BB_Pos", 0.5)
    if 0.3 <= bb_pos <= 0.6:
        score += 1
        reasons.append("布林帶位置健康，非超買區")
    elif bb_pos > 0.85:
        score -= 1
        warnings.append("接近布林帶上緣，追高風險")
    elif bb_pos < 0.2:
        warnings.append("接近布林帶下緣，方向不明")

    # ── 突破 ─────────────────────────────────────────────
    if price > recent_high:
        score += 2
        reasons.append("價格突破近 20 日高點")
        setup_type = "Breakout"
    elif price < recent_low:
        score -= 2
        warnings.append("價格跌破近 20 日低點")

    # ── 4H / 1H 多框架 ──────────────────────────────────
    if h4_last["Close"] > h4_last["EMA20"]:
        score += 1
        reasons.append("4H 仍站在 EMA20 上方")
    else:
        warnings.append("4H 尚未重新站穩 EMA20")

    if not h4_last["Close"] > h4_last.get("EMA50", h4_last["EMA20"]):
        score -= 1
        warnings.append("4H 跌破 EMA50，中期趨勢不夠強")

    if h1_last["Close"] > h1_last["EMA20"]:
        score += 1
        reasons.append("1H 短線進場結構偏強")
    else:
        warnings.append("1H 進場點還不夠漂亮")

    if h1_last.get("MACD_Hist", 0) > 0:
        score += 1
        reasons.append("1H MACD 動能翻正，進場確認度高")
    else:
        warnings.append("1H MACD 動能尚未翻正，等待確認")

    # 三框架全偏弱 → 嚴格禁止
    timeframe_alignment = sum([
        price > ema20 > ema50,
        h4_last["Close"] > h4_last["EMA20"],
        h1_last["Close"] > h1_last["EMA20"],
    ])
    if timeframe_alignment == 0:
        hard_no_trade = True
        warnings.append("三個時間框架全部偏弱，嚴格禁止進場")

    # ══════════════════════════════════════════════════════
    # 進場區間
    # ══════════════════════════════════════════════════════
    entry_zone_low  = ema20 - atr * 0.3
    entry_zone_high = ema20 + atr * 0.3
    planned_entry   = entry_zone_high

    # 動態止損（原有）
    if price > recent_high:
        stop_loss = entry_zone_low - atr * 0.5
    else:
        stop_loss = entry_zone_low - atr * 0.8

    # ══════════════════════════════════════════════════════
    # 鐵律三：止損絕對執行（2*ATR below entry，永遠輸出）
    # ══════════════════════════════════════════════════════
    stop_loss_price = round(planned_entry - 2 * atr, 2)

    target_1 = entry_zone_high + atr * 2.0
    target_2 = entry_zone_high + atr * 3.5

    risk     = planned_entry - stop_loss
    reward   = target_1 - planned_entry
    rr_ratio = reward / risk if risk > 0 else 0
    distance_from_entry = (price - planned_entry) / planned_entry

    if rr_ratio < 2.0:
        hard_no_trade = True
        warnings.append(f"風報比 {rr_ratio:.1f}R 不足 2.0R，不值得冒這個風險")

    if rsi > 75:
        hard_no_trade = True
        warnings.append("RSI 過熱，容易追在短線高點")

    # ── Setup 類型最終判斷 ────────────────────────────────
    if abs(price - ema20) / ema20 < 0.02:
        setup_type = "Pullback"
    elif price > recent_high:
        setup_type = "Breakout"

    if distance_from_entry > 0.05:
        if price > recent_high and vol_ratio > 1.5 and market["score"] >= 2:
            reasons.append("強勢突破結構，允許 Momentum Breakout 模式")
            setup_type = "Momentum Breakout"
        else:
            hard_no_trade = True
            warnings.append("現價距離等待區過遠，不適合追價")
            setup_type = "Overextended"

    # ══════════════════════════════════════════════════════
    # 出場分析一：動態防守價（Trailing Stop，移動停利）
    # ══════════════════════════════════════════════════════
    recent_peak_close  = daily["Close"].iloc[-60:].max()
    trailing_stop_3atr = round(recent_peak_close - 3 * atr, 2)
    trailing_stop_ma5  = round(float(ema5), 2)
    trailing_stop      = max(trailing_stop_3atr, trailing_stop_ma5)
    holding_ok         = price > trailing_stop

    # 預設：安心抱
    exit_action  = "hold"
    exit_message = (
        f"✅【安心抱緊緊，先不要賣！】\n"
        f"大戶還在踩油門，只要沒跌破 {trailing_stop:.2f} 元就安心抱著，"
        "讓子彈飛，我們一起賺更多！"
    )

    # 跌破防線 → 提醒檢視
    if not holding_ok:
        exit_action  = "review"
        exit_message = (
            f"⚠️【注意！今日跌破動態防線 {trailing_stop:.2f}】\n"
            "建議認真考慮減倉，保護到手的利潤。"
        )

    # ══════════════════════════════════════════════════════
    # 出場分析二：極端正乖離 > 30%（分批停利）
    # ══════════════════════════════════════════════════════
    if deviation_from_ma20_pct > 30:
        exit_action  = "partial_exit"
        exit_message = (
            f"🟡【金蟬脫殼，先賣 1/3 把錢放口袋！】\n"
            f"正乖離 {deviation_from_ma20_pct:.1f}%，市場陷入散戶瘋狂期！"
            "阿公阿嬤聽話，先賣掉三分之一，把賺來的錢放口袋，"
            "剩下的繼續看它飛，這樣晚上才睡得著覺！"
        )

    # ══════════════════════════════════════════════════════
    # 出場分析三：高檔爆量不漲（大戶倒貨警報）
    # ══════════════════════════════════════════════════════
    candle_range        = (high_price - low_price) if high_price != low_price else 0.01
    upper_shadow        = high_price - max(d["Close"], open_price)
    upper_shadow_ratio  = upper_shadow / candle_range
    is_red_candle       = d["Close"] < open_price

    if vol_ratio > 3.0 and (upper_shadow_ratio > 0.3 or is_red_candle):
        distribution_alert = True
        exit_action  = "full_exit"
        exit_message = (
            f"🔴【警報！大戶在跑了，我們全拿現！】\n"
            f"成交量爆到平均的 {vol_ratio:.1f} 倍，K 線出現"
            f"{'長上影線' if upper_shadow_ratio > 0.3 else '收黑K'}，"
            "大老闆們在偷偷坐電梯下樓了！"
            "今天請把股票全部賣掉，保住所有利潤，獲利落袋為安！"
        )

    # ── 評級系統 ─────────────────────────────────────────
    if storm_mode:
        rating = "🔴 Storm Defense / 風暴防禦"
        bias   = "防禦現金"
    elif overheat_alert:
        rating = "🔴 Overheat / 追高熔斷"
        bias   = "等候拉回"
    elif hard_no_trade and setup_type != "Momentum Breakout":
        rating = "🔴 No Trade"
        bias   = "風險過高"
    elif setup_type == "Momentum Breakout":
        rating = "🟡 Momentum Watchlist"
        bias   = "強勢突破"
    elif score >= 6:
        rating = "🟢 High Quality Setup"
        bias   = "偏多"
    elif score >= 3:
        rating = "🟡 Watchlist / 等回踩"
        bias   = "偏多但需要確認"
    elif score <= -3:
        rating = "🔴 Avoid / 偏空"
        bias   = "偏空"
    else:
        rating = "🔴 No Trade"
        bias   = "方向不明"

    return {
        # ── 基本技術指標 ──
        "price":        price,
        "ema20":        ema20,
        "ema50":        ema50,
        "rsi":          rsi,
        "atr":          atr,
        "vol_ratio":    vol_ratio,
        "recent_high":  recent_high,
        "recent_low":   recent_low,
        # ── 評級 & 系統 ──
        "score":        score,
        "rating":       rating,
        "bias":         bias,
        "reasons":      reasons,
        "warnings":     warnings,
        "setup_type":   setup_type,
        "market_status": market["status"],
        # ── 進場計畫 ──
        "entry_zone_low":     entry_zone_low,
        "entry_zone_high":    entry_zone_high,
        "planned_entry":      planned_entry,
        "distance_from_entry": distance_from_entry,
        # ── 鐵律三：止損（雙版本）──
        "stop_loss":       stop_loss,
        "stop_loss_price": stop_loss_price,   # 絕對止損 = entry - 2*ATR
        # ── 目標 & 風報比 ──
        "target_1":  target_1,
        "target_2":  target_2,
        "rr_ratio":  rr_ratio,
        # ── 鐵律一：追高防護 ──
        "deviation_from_ma20_pct": round(deviation_from_ma20_pct, 1),
        "overheat_alert":          overheat_alert,
        # ── 鐵律二：風暴防禦 ──
        "storm_mode": storm_mode,
        # ── 出場分析 ──
        "trailing_stop":       trailing_stop,
        "exit_action":         exit_action,
        "exit_message":        exit_message,
        "distribution_alert":  distribution_alert,
    }


# =========================
# 部位計算器
# =========================

def calc_position_size(setup: dict, account_size: float, risk_pct: float = 0.02) -> dict:
    """
    依帳戶規模與 2% 風控規則計算合理部位大小。
    Returns dict with shares, position_value, max_loss etc.
    """
    planned_entry   = setup.get("planned_entry", 0)
    stop_loss_price = setup.get("stop_loss_price") or setup.get("stop_loss", 0)

    if planned_entry <= 0 or stop_loss_price <= 0:
        return {"error": "無法計算：進場價或止損價為零"}

    risk_per_share = planned_entry - stop_loss_price
    if risk_per_share <= 0:
        return {"error": "止損價必須低於進場價，請確認 setup 是否有效"}

    risk_budget        = account_size * risk_pct           # 這筆最多虧多少錢
    shares_by_risk     = risk_budget / risk_per_share      # 按風險反推股數
    max_position_value = account_size * 0.20               # 單一部位上限 20%
    shares_by_conc     = max_position_value / planned_entry

    shares         = int(min(shares_by_risk, shares_by_conc))
    shares         = max(shares, 1)
    position_value = shares * planned_entry
    actual_loss    = shares * risk_per_share

    return {
        "shares":          shares,
        "planned_entry":   planned_entry,
        "stop_loss_price": stop_loss_price,
        "risk_per_share":  round(risk_per_share, 2),
        "position_value":  round(position_value, 2),
        "max_loss":        round(actual_loss, 2),
        "pct_of_account":  round(position_value / account_size * 100, 1),
        "risk_pct_actual": round(actual_loss / account_size * 100, 2),
        "account_size":    account_size,
    }


def format_position_size(symbol: str, size: dict) -> str:
    if "error" in size:
        return f"❌ 部位計算失敗：{size['error']}"
    return (
        f"💰 {symbol} 部位計算\n"
        f"帳戶規模：${size['account_size']:,.0f}\n\n"
        f"建議買入：{size['shares']} 股\n"
        f"計畫進場：${size['planned_entry']:,.2f}\n"
        f"投入金額：${size['position_value']:,.2f}（佔帳戶 {size['pct_of_account']:.1f}%）\n\n"
        f"🛡️ 絕對止損：${size['stop_loss_price']:,.2f}\n"
        f"每股風險：${size['risk_per_share']:,.2f}\n"
        f"最大損失：${size['max_loss']:,.2f}（帳戶 {size['risk_pct_actual']:.2f}%）\n\n"
        f"⚠️ 這是 2% 風控計算，不保證獲利。\n"
        f"跌破 ${size['stop_loss_price']:,.2f} 請一定要果斷賣掉，這是在保護我們的退休金！"
    )


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
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol":       symbol,
        "setup_type":   setup.get("setup_type"),
        "rating":       setup.get("rating"),
        "market_status": setup.get("market_status"),
        "price":        round(setup.get("price", 0), 2),
        "planned_entry": round(setup.get("planned_entry", 0), 2),
        "stop_loss":    round(setup.get("stop_loss", 0), 2),
        "stop_loss_price": round(setup.get("stop_loss_price", 0), 2),
        "trailing_stop": round(setup.get("trailing_stop", 0), 2),
        "target_1":     round(setup.get("target_1", 0), 2),
        "target_2":     round(setup.get("target_2", 0), 2),
        "rr_ratio":     round(setup.get("rr_ratio", 0), 2),
        "deviation_from_ma20_pct": round(setup.get("deviation_from_ma20_pct", 0), 1),
        "overheat_alert":    setup.get("overheat_alert", False),
        "storm_mode":        setup.get("storm_mode", False),
        "distribution_alert": setup.get("distribution_alert", False),
        "exit_action":       setup.get("exit_action", "hold"),
    }

    trades.append(trade_data)

    with open("trades.json", "w") as f:
        json.dump(trades, f, indent=4)

    print(f"📝 已紀錄 setup: {symbol}")

def analyze_trade_history():
    try:
        with open("trades.json", "r") as f:
            trades = json.load(f)
    except Exception as e:
        return f"無交易記錄或文件讀取失敗: {str(e)}"

    if not isinstance(trades, list):
        return "交易記錄格式有誤"

    if len(trades) < 3:
        return f"目前共記錄 {len(trades)} 筆，資料不足以分析（需要至少 3 筆）"

    try:
        df = pd.DataFrame(trades)

        setup_counts = df["setup_type"].value_counts().to_dict()
        rating_counts = df["rating"].value_counts().to_dict()

        recent = df.tail(20)
        market_dist = recent["market_status"].value_counts().to_dict()

        avg_rr = df["rr_ratio"].mean()

        lines = ["📊 交易記錄分析"]
        lines.append(f"總記錄筆數：{len(trades)}")
        lines.append(f"平均風報比：{avg_rr:.2f}R")

        # ── 真實勝率（由 scheduler 每日結算）────────────────
        settled  = [t for t in trades if t.get("outcome") in ("win", "loss")]
        timeouts = [t for t in trades if t.get("outcome") == "timeout"]
        pending  = [t for t in trades if not t.get("outcome")]

        lines.append(f"\n🎯 實際勝率分析：")
        if settled:
            wins     = sum(1 for t in settled if t["outcome"] == "win")
            losses   = len(settled) - wins
            win_rate = wins / len(settled) * 100
            lines.append(f"  已結算：{len(settled)} 筆（勝 {wins} | 敗 {losses}）")
            lines.append(f"  實際勝率：{win_rate:.1f}%")
        else:
            lines.append("  尚無已結算的交易（每日盤後自動計算）")
        if timeouts:
            lines.append(f"  超時未觸發：{len(timeouts)} 筆（未進場）")
        if pending:
            lines.append(f"  待結算：{len(pending)} 筆")

        lines.append(f"\nSetup 分佈：")
        for k, v in setup_counts.items():
            lines.append(f"  {k}: {v} 次")
        lines.append(f"\n系統評級分佈：")
        for k, v in rating_counts.items():
            lines.append(f"  {k}: {v} 次")
        lines.append(f"\n最近 20 筆大盤環境：")
        for k, v in market_dist.items():
            lines.append(f"  {k}: {v} 次")

        return "\n".join(lines)
    except Exception as e:
        return f"分析失敗: {str(e)}"

def is_stock_related(text):
    """檢查問題是否與股票相關"""
    stock_keywords = [
        "股票", "漲", "跌", "買", "賣", "交易", "投資", "ETF", "基金",
        "產業", "技術", "分析", "圖表", "走勢", "K線", "均線", "RSI",
        "MACD", "成交量", "籌碼", "融資", "賺", "虧", "獲利", "停損",
        "目標", "支撐", "壓力", "突破", "趨勢", "波段", "短線", "長線",
        "套住", "解套", "反彈", "回檔", "盤整", "強勢", "弱勢", "大漲",
        "閃崩", "融券", "借券", "除權", "除息", "配股", "配息", "新股",
        "IPO", "下市", "上市", "公告", "財報", "營收", "EPS", "本益比",
        "股價", "股數", "市值", "成交", "掛單", "委買", "委賣", "量能",
        "波動", "風險", "報酬", "風報", "進場", "出場", "加碼", "減碼",
        "美股", "台股", "美國股市", "台灣股市", "推薦", "建議", "挑選",
        "選股", "幫我", "掃描", "哪些", "什麼股", "哪支", "適合", "能買",
        "可以買", "值得買", "現在買", "掃一下", "scan", "screen",
    ]

    text_lower = text.lower()

    # 如果包含股票關鍵字
    if any(kw in text_lower for kw in stock_keywords):
        return True

    # 如果是股票代號或台股代號
    if re.search(r"[A-Z]{1,5}|\d{4}", text):
        return True

    return False


def is_direct_ticker(text):
    text = text.strip().upper()
    # 排除帶斜杠的命令
    if text.startswith("/"):
        return False
    # 排除命令關鍵字
    if text in ["STATUS", "STATS", "HELP", "MORNING"]:
        return False
    return bool(re.fullmatch(r"[A-Z]{1,5}", text)) or bool(re.fullmatch(r"\d{4}", text))


def extract_symbol_from_text(text):
    text_upper = text.upper()

    common_symbols = [
        "TSLA", "NVDA", "AAPL", "AMD", "MU", "MSFT", "META",
        "AMZN", "GOOGL", "GOOG", "QQQ", "SPY", "PLTR",
        "SMCI", "COIN", "MSTR", "NFLX", "AVGO"
    ]

    for symbol in common_symbols:
        if symbol in text_upper:
            return symbol

    match = re.search(r"\b[A-Z]{2,5}\b", text_upper)
    if match:
        return match.group(0)

    match_tw = re.search(r"\b\d{4}\b", text)
    if match_tw:
        return match_tw.group(0)

    return None


def get_stock_snapshot(symbol):
    try:
        result = _router.route(symbol)

        if result.daily.empty or result.h4.empty or result.h1.empty:
            return None, f"找不到 {symbol} 的足夠資料。"

        daily = add_indicators(result.daily).dropna()
        h4 = add_indicators(result.h4).dropna()
        h1 = add_indicators(result.h1).dropna()

        market = get_market_filter()
        setup = detect_swing_setup(daily, h4, h1, market)
        news = get_stock_news(result.symbol)

        return {
            "symbol": result.symbol,
            "setup": setup,
            "news": news,
            "market": market
        }, None

    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)

_US_SCAN_LIST = [
    "NVDA", "TSLA", "AAPL", "AMD", "META", "MSFT", "AMZN",
    "PLTR", "AVGO", "SMCI", "COIN", "NFLX", "NOW", "CRWD",
    "PANW", "MSTR", "GOOGL", "UBER", "ARM",
]
_TW_SCAN_LIST = ["2330", "2317", "2454", "2308", "3008", "2382", "2412", "2303"]


def scan_market_sync(market: str = "US") -> list[dict]:
    """
    同步掃描 watchlist，回傳有進場機會（High / Watch）的 setup 清單。
    設計為在 run_in_executor 中呼叫。
    """
    watchlist = _US_SCAN_LIST if market == "US" else _TW_SCAN_LIST
    results: list[dict] = []

    mkt_filter = get_market_filter()

    for symbol in watchlist:
        try:
            result = _router.route(symbol)
            if result.daily.empty or result.h4.empty or result.h1.empty:
                continue
            daily = add_indicators(result.daily).dropna()
            h4    = add_indicators(result.h4).dropna()
            h1    = add_indicators(result.h1).dropna()
            if len(daily) < 20:
                continue
            setup = detect_swing_setup(daily, h4, h1, mkt_filter)
            rating = setup.get("rating", "").lower()
            if rating.startswith("high") or "watch" in rating:
                results.append({
                    "symbol":        result.symbol,
                    "rating":        setup["rating"],
                    "rr_ratio":      setup.get("rr_ratio", 0),
                    "setup_type":    setup.get("setup_type", "N/A"),
                    "planned_entry": setup.get("planned_entry", 0),
                    "stop_loss":     setup.get("stop_loss", 0),
                    "target_1":      setup.get("target_1", 0),
                })
        except Exception:
            continue

    results.sort(key=lambda x: x["rr_ratio"], reverse=True)
    return results[:6]


def is_industry_question(text):
    keywords = [
        "產業", "趨勢", "資金", "流向", "區域", "板塊",
        "優勢", "強勢族群", "題材", "半導體", "AI",
        "電動車", "雲端", "能源", "金融", "台積電"
    ]
    return any(k in text for k in keywords)


def get_industry_chat_response(user_msg):
    industry_map = {
        "半導體": ["NVDA", "AMD", "MU", "TSM", "AVGO"],
        "AI": ["NVDA", "MSFT", "GOOGL", "META", "AVGO"],
        "電動車": ["TSLA", "RIVN", "LI", "XPEV"],
        "雲端": ["MSFT", "AMZN", "GOOGL"],
        "金融": ["JPM", "BAC", "GS"],
        "能源": ["XOM", "CVX", "OXY"],
        "台積電": ["TSM", "NVDA", "AMD", "AVGO"],
    }

    selected_theme = None

    for theme in industry_map:
        if theme in user_msg:
            selected_theme = theme
            break

    if not selected_theme:
        selected_theme = "AI"

    symbols = industry_map[selected_theme]
    market = get_market_filter()
    summaries = []

    for symbol in symbols:
        try:
            snapshot, error = get_stock_snapshot(symbol)

            if snapshot:
                setup = snapshot["setup"]
                summaries.append(
                    f"{symbol}: {setup['rating']} | {setup['setup_type']} | "
                    f"現價 {setup['price']:.2f} | RSI {setup['rsi']:.1f} | "
                    f"偏離等待區 {setup['distance_from_entry'] * 100:.1f}%"
                )
            else:
                summaries.append(f"{symbol}: 資料不足")

        except Exception as e:
            summaries.append(f"{symbol}: 分析失敗 {str(e)}")

    prompt = f"""
你是 WengStock AI，一個專業但好聊天的股票與產業趨勢助理。

使用者問題：
{user_msg}

目前分析主題：
{selected_theme}

大盤狀態：
{market}

代表股票狀態：
{summaries}

請用繁體中文回答。

回答目標：
- 不要只講單一股票
- 要回答產業趨勢、資金是否偏向流入、目前強弱
- 用人話講，不要像研究報告
- 如果族群已經漲太多，要提醒不要追高
- 如果產業還強，但位置不好，要說「產業強，不代表現在每一檔都能追」
- 可以用台積電、NVDA、AMD、MU 這種代表股來解釋
- 不要保證獲利
- 不要亂編沒有提供的資料

格式：
第一段：直接回答這個產業/主題目前強不強。
第二段：講資金/代表股狀況。
第三段：講現在操作上該怎麼看。
最後一句：風控提醒。

不要超過 8 句。
像 LINE 上交易員回朋友。
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.45,
        max_tokens=700,
        messages=[
            {
                "role": "system",
                "content": "你是 WengStock AI，專業但像真人聊天的產業趨勢與股票交易助理。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

def get_stock_chat_response(user_msg):
    symbol = extract_symbol_from_text(user_msg)

    context_text = ""

    if symbol:
        snapshot, error = get_stock_snapshot(symbol)

        if snapshot:
            setup = snapshot["setup"]
            news_text = "\n".join([f"- {n['title']}" for n in snapshot["news"][:3]])

            context_text = f"""
使用者提到股票：{snapshot["symbol"]}

目前數據：
現價：{setup["price"]:.2f}
EMA20：{setup["ema20"]:.2f}
EMA50：{setup["ema50"]:.2f}
RSI：{setup["rsi"]:.2f}
ATR：{setup["atr"]:.2f}
量比：{setup["vol_ratio"]:.2f}
系統評級：{setup["rating"]}
Setup 類型：{setup["setup_type"]}
方向：{setup["bias"]}
大盤狀態：{setup["market_status"]}
等待區：{setup["entry_zone_low"]:.2f} - {setup["entry_zone_high"]:.2f}
停損：{setup["stop_loss"]:.2f}
目標一：{setup["target_1"]:.2f}
風報比：{setup["rr_ratio"]:.2f}R
偏離等待區：{setup["distance_from_entry"] * 100:.2f}%

支持理由：
{setup["reasons"]}

風險警告：
{setup["warnings"]}

近期新聞：
{news_text}
"""
        else:
            context_text = f"使用者提到 {symbol}，但資料抓取失敗：{error}"

    else:
        market = get_market_filter()
        context_text = f"""
使用者沒有明確提到股票代號。

目前大盤狀態：
{market}
"""

    prompt = f"""
你是 WengStock AI，一個股票交易聊天助理。

你的定位：
- 不是喊單老師
- 不保證獲利
- 幫使用者判斷風險、位置、進場是否合理
你不是分析報告生成器。

你是有多年經驗的美股 swing trader，
平常會直接跟朋友聊股票。

回答風格：

- 短
- 直接
- 有交易員感
- 不要每次都用 1. 2. 3. 條列
- 更像真人在 LINE 回朋友
- 可以短段落回答
- 語氣要像交易員，不要像客服
- 可以適度使用：「老實說」、「這位置」、「我會等」、「這種我不追」、「有點像 FOMO」
- 如果風險高，第一句就直接講不要追
- 如果使用者被套住，先安撫，再講處理方式
- 不要只丟指標，要翻譯成人話
- 不要像 AI
- 不要像新聞稿
- 不要像技術分析文章
- 不要每句都提 RSI / EMA
- 用「這位置我不會追」、「我會等回踩」、「風險報酬不漂亮」這種口語

你的任務不是炫技術分析，
而是幫使用者避開爛交易。

如果 setup 很差，
直接說不要做。

如果位置太高，
直接說不要追。

如果是好 setup，
也不要過度興奮。

像真正職業交易員聊天。
- 回答要短、直接、有交易員感
- 使用繁體中文
- 如果使用者想追高，要直接提醒風險
- 如果資料不足，要誠實說資料不足
- 不要亂編沒有提供的數據

使用者問題：
{user_msg}

可用資料：
{context_text}

請用「聊天回覆」格式回答，不要用完整 8 點報告。

格式：
像 LINE 聊天一樣回答。

第一段：直接講結論。
如果使用者明顯有情緒：
例如：
- 怕錯過
- 被套住
- 很想追
- 很焦慮

先像真人一樣回應情緒，
再分析。

不要直接冷冰冰開始技術分析。
不要像老師上課。

更像：
一個有經驗的 trader，
在 LINE 回朋友。
第二段：用人話解釋原因，不要堆指標。
第三段：給下一步行動。

不要超過 8 句。
除非使用者要求完整報告，否則不要條列太多。
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.45,
        max_tokens=600,
        messages=[
            {
                "role": "system",
                "content": """你是 WengStock AI，專業的美股 Swing Trading 交易助理。

你的定位：
- 只回答股票和交易相關的問題
- 有 10+ 年經驗的職業交易員
- 重視風控和交易紀律，不是喊單老師
- 用人話講，像真人交易員在 LINE 聊天

你的原則：
1. 第一優先是幫使用者避開爛交易，不是鼓勵做交易
2. 如果風險高或位置不好，直接說不要做
3. 如果資料不足，誠實說資料不足，不要亂編
4. 不保證獲利，不說穩賺
5. 優先提醒風險，再講機會

回答風格：
- 短、直接、有交易員感
- 用「這位置我不會追」「我會等回踩」「風險報酬不漂亮」這種口語
- 不要條列 1. 2. 3.（除非必要）
- 更像 LINE 上的真人聊天，不像研究報告
- 可以用「老實說」「有點像 FOMO」「這種我不做」

如果使用者提到股票代號，利用提供的技術數據給出專業判斷。
如果使用者沒提股票，用大盤狀態給出市場觀察。"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

def get_ai_analysis(user_input):
    try:
        user_input = user_input.strip()

        if user_input.startswith("/") or not user_input or len(user_input) > 10:
            return f"❌ '{user_input}' 不是有效的股票代號，請輸入正確的代號（例如：NVDA, TSLA, 2330）"

        try:
            result = _router.route(user_input)
        except ValueError:
            return f"❌ '{user_input}' 不是有效的股票代號，請輸入正確的代號（例如：NVDA, TSLA, 2330）"

        symbol = result.symbol
        print(f"🚀 Swing 分析中: {symbol} [{result.market.value}]")

        if result.daily.empty or result.h4.empty or result.h1.empty:
            return f"❌ 找不到 {symbol} 的足夠資料，請檢查代號。"

        daily = add_indicators(result.daily).dropna()
        h4 = add_indicators(result.h4).dropna()
        h1 = add_indicators(result.h1).dropna()

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
    news_text = "\n".join([f"- {n['title']}" for n in news])

    # ── 六條防護邏輯狀態 ────────────────────────────────────
    storm_mode        = setup.get("storm_mode", False)
    overheat_alert    = setup.get("overheat_alert", False)
    distribution_alert = setup.get("distribution_alert", False)
    exit_action       = setup.get("exit_action", "hold")
    exit_message      = setup.get("exit_message", "")
    dev_pct           = setup.get("deviation_from_ma20_pct", 0)
    trailing_stop     = setup.get("trailing_stop", setup["stop_loss"])
    stop_loss_price   = setup.get("stop_loss_price", setup["stop_loss"])

    # ── 特殊情境的強制指示段 ─────────────────────────────────
    iron_rule_section = ""

    if storm_mode:
        iron_rule_section += f"""
⛔⛔⛔ 【大盤風暴防禦模式啟動】⛔⛔⛔
{setup.get('warnings', [''])[0] if setup.get('warnings') else ''}
你必須用最溫柔但堅定的語氣，用大白話（像對阿公阿嬤說話），
完整傳達以下訊息：「現在市場正在刮颱風，外面很危險，
AI 建議您現在抱著現金好好休息，等大晴天我們再出來開工！」
不要做任何進場分析，只做防禦建議。
"""

    if overheat_alert and not storm_mode:
        iron_rule_section += f"""
🔥 【追高強行熔斷】
正乖離率 {dev_pct:.1f}%，已超過 15% 安全門檻。
你必須用最溫柔但堅定的語氣，用大白話告訴用戶：
「現在這檔股票太熱太貴了，我們去買會幫別人洗碗，請耐心等它降溫拉回再考慮！」
"""

    if distribution_alert:
        iron_rule_section += f"""
🚨 【大戶倒貨警報】
高檔爆量 {setup['vol_ratio']:.1f}x + K線異常，出場訊號已觸發。
請完整傳達以下大白話訊息並放在最顯眼位置：
{exit_message}
"""
    elif exit_action in ("partial_exit", "review"):
        iron_rule_section += f"""
📤 【出場訊號】{exit_action}
{exit_message}
請在報告中清楚傳達上述出場建議。
"""

    prompt = f"""
你是 WengStock AI，一個充滿良心、最重視保護散戶與長輩資產的 Swing Trading 分析助理。
你的核心宗旨：高勝率、絕對不割韭菜、全心保護散戶與長輩的長線資產。

{iron_rule_section}

股票：{symbol}

═══════════ 技術數據 ═══════════
現價：{setup["price"]:.2f}
EMA20：{setup["ema20"]:.2f}  EMA50：{setup["ema50"]:.2f}
RSI：{setup["rsi"]:.2f}  ATR：{setup["atr"]:.2f}  量比：{setup["vol_ratio"]:.2f}
正乖離率（MA20）：{dev_pct:.1f}%
系統分數：{setup["score"]}  評級：{setup["rating"]}
Setup 類型：{setup["setup_type"]}  方向：{setup["bias"]}
大盤狀態：{setup["market_status"]}

═══════════ 進場計畫 ═══════════
等待回踩區：{setup["entry_zone_low"]:.2f} - {setup["entry_zone_high"]:.2f}
計畫進場價：{setup["planned_entry"]:.2f}
偏離等待區：{setup["distance_from_entry"] * 100:.1f}%

═══════════ 鐵律三：止損（必須輸出）═══════════
絕對止損價（2ATR below entry）：{stop_loss_price:.2f}
動態止損參考：{setup["stop_loss"]:.2f}

═══════════ 目標 & 風報比 ═══════════
第一目標：{setup["target_1"]:.2f}  第二目標：{setup["target_2"]:.2f}
風報比：{setup["rr_ratio"]:.2f}R

═══════════ 出場分析 ═══════════
動態防守價（Trailing Stop）：{trailing_stop:.2f}
出場動作：{exit_action}
出場訊息：{exit_message}

═══════════ 支持理由 / 風險警告 ═══════════
支持理由：{setup["reasons"]}
風險警告：{setup["warnings"]}

═══════════ 近期新聞 ═══════════
{news_text}

═══════════ 輸出要求 ═══════════

請用繁體中文，輸出 LINE 適合閱讀的格式：

📊 {symbol} Swing 分析

1. 【方向 & 現況】目前趨勢與大盤評估
2. 【Setup 品質】分析評級原因，如果 No Trade 說清楚為什麼
3. 【進場策略】等待區、是否適合追價、距離評估
4. 【🛡️ 止損鐵律】
   必須用堅定的大白話寫這一段：
   「看錯不可恥，如果跌破 {stop_loss_price:.2f} 元，請一定要果斷賣掉，
   這是在保護我們的退休金！」
5. 【出場訊號】
   根據 exit_action 輸出對應的大白話出場建議（已在上面提供，直接套用）
6. 【新聞面】新聞是否真的重要，對 swing 有無影響
7. 【最後一句】一句簡短的交易紀律提醒

絕對規則：
- 不要保證獲利，不要說穩賺
- 如果 storm_mode 或 overheat_alert 啟動，第一段必須用大白話傳達熔斷訊息
- 止損那段語氣必須溫柔但堅定，像在保護長輩
- 不要加入 warnings 裡沒有的風險
- 如果評級是 🔴，不要強調目標價，說「目前不執行目標，先等位置回到等待區」
- 如果 distribution_alert 觸發，出場建議放在最顯眼位置（最前面或最大字）
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.35,
        max_tokens=800,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是 WengStock AI，一個充滿良心的 Swing Trading 分析助理。"
                    "你的核心宗旨是保護散戶與長輩資產，不割韭菜，永遠風控優先。"
                    "你說話像一個有經驗的交易員在 LINE 上跟長輩解釋，溫柔但堅定，人話不是術語。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
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

    try:
        with open("/tmp/agent_webhook.log", "a") as f:
            f.write(f"[reply_line] Sending to token: {reply_token[:20]}..., text_len: {len(text)}\n")
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        with open("/tmp/agent_webhook.log", "a") as f:
            f.write(f"[reply_line] Status: {r.status_code}, Response: {r.text[:200]}\n")
        print(f"✅ LINE reply status: {r.status_code}")
        if r.status_code != 200:
            print(f"❌ LINE reply error: {r.text}")
    except Exception as e:
        with open("/tmp/agent_webhook.log", "a") as f:
            f.write(f"[reply_line] Exception: {str(e)}\n")
        print(f"❌ LINE reply exception: {str(e)}")

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
    with open("/tmp/agent_webhook.log", "a") as f:
        f.write(f"\n{'='*50}\n[{datetime.now()}] BODY: {body}\n")
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
🤖 WengStock AI 使用指南

━━━━━━━━━━━━━━━━━━━━━━━━━
📊 【完整 Swing 分析】
━━━━━━━━━━━━━━━━━━━━━━━━━
直接輸入股票代號：
• NVDA
• TSLA
• AMD
• 2330（台股）

你會得到：
✓ 技術評級（🟢 High Quality / 🟡 Watchlist / 🔴 No Trade）
✓ Setup 類型（Breakout / Pullback / Momentum Breakout）
✓ 技術指標（RSI、MACD、布林帶、量能）
✓ 等待區 & 進場點
✓ 停損位置 & 目標價
✓ 風報比（R值，越高越優）
✓ 近期新聞面

━━━━━━━━━━━━━━━━━━━━━━━━━
💬 【聊天模式】
━━━━━━━━━━━━━━━━━━━━━━━━━
像真人交易員一樣問：
• TSLA 現在能追嗎？
• NVDA 跌到哪裡可以接？
• AMD 我已經買了，要不要停損？
• 今天適合進攻還是保守？
• 大盤怎麼看？

（系統會根據最新技術數據回答）

━━━━━━━━━━━━━━━━━━━━━━━━━
🌎 【產業 & 資金面】
━━━━━━━━━━━━━━━━━━━━━━━━━
問產業趨勢：
• 半導體現在還強嗎？
• AI 族群有人搶籌嗎？
• 電動車資金怎麼走？
• 雲端概念誰最強？

（回答產業整體方向，不是單一股票）

━━━━━━━━━━━━━━━━━━━━━━━━━
📈 【快速命令】
━━━━━━━━━━━━━━━━━━━━━━━━━
/exit NVDA 出場分析（手上有股票要不要賣）
/morning   盤前簡報（美股開盤前掃描）
/stats     交易統計（系統勝率分析）
/help      這份說明

━━━━━━━━━━━━━━━━━━━━━━━━━
⚙️ 【技術指標說明】
━━━━━━━━━━━━━━━━━━━━━━━━━
• EMA20/50：趨勢方向
• RSI：超買超賣（>75 過熱，<40 無力）
• MACD：動能方向（柱狀圖翻正 = 轉強）
• ATR：波動度（用於止損、目標計算）
• 布林帶：超買超賣區間
• 量比：買賣力道是否充足

━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 【Setup 類型】
━━━━━━━━━━━━━━━━━━━━━━━━━
Breakout     突破模式（衝破近期高點）
Pullback     回踩模式（下探後反彈）
Momentum     強勢追漲（需要極強大盤）
No Trade     不執行（位置差 / 風險大）

━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 【重要提醒】
━━━━━━━━━━━━━━━━━━━━━━━━━
✗ 我不是喊單工具
✗ 我不保證獲利
✓ 我幫你看風險和位置
✓ 我提醒不要追爛交易
✓ 我關注風險報酬比
"""
                reply_line(reply_token, help_text)

            elif user_msg.lower().strip().startswith("/exit ") or user_msg.lower().strip().startswith("exit "):
                parts = user_msg.strip().split()
                if len(parts) >= 2 and is_direct_ticker(parts[1]):
                    exit_sym = parts[1].upper()
                    reply_line(reply_token, f"📤 {exit_sym} 出場分析中，請稍候約 15 秒...")
                    # Flask 版沒有 BackgroundTasks，直接同步跑（未來遷移到 main.py 異步版）
                    try:
                        snapshot, err = get_stock_snapshot(exit_sym)
                        if snapshot:
                            s = snapshot["setup"]
                            exit_msg = s.get("exit_message", "")
                            trailing = s.get("trailing_stop", 0)
                            slp = s.get("stop_loss_price", s["stop_loss"])
                            reply_line(reply_token,
                                f"📤 {exit_sym} 出場分析\n\n"
                                f"動態防守價：{trailing:.2f}\n"
                                f"絕對止損：{slp:.2f}\n\n"
                                f"{exit_msg}\n\n"
                                f"⚠️ 看錯不可恥，跌破止損請果斷賣掉，保護我們的退休金！"
                            )
                        else:
                            reply_line(reply_token, f"❌ {err}")
                    except Exception as e:
                        reply_line(reply_token, f"⚠️ 出場分析失敗：{str(e)}")

            elif user_msg.lower().strip() in ["/stats", "stats", "統計"]:
                try:
                    stats = analyze_trade_history()
                    reply_line(reply_token, stats)
                except Exception as e:
                    reply_line(reply_token, f"⚠️ /stats 錯誤: {str(e)}")

            else:
                # 先檢查是否與股票相關
                if not is_stock_related(user_msg):
                    reply_line(
                        reply_token,
                        "我是 Swing Trading 專家，只能回答股票相關的問題 📈\n\n"
                        "你可以問我：\n"
                        "• 某支股票的技術分析\n"
                        "• 產業趨勢和資金流向\n"
                        "• 交易策略和風控\n"
                        "• 市場行情解讀\n\n"
                        "或輸入 /help 查看更多功能"
                    )
                elif is_direct_ticker(user_msg):
                    analysis = get_ai_analysis(user_msg)
                    reply_line(
                        reply_token,
                        f"📊 {user_msg.upper()} Swing 分析報告：\n\n{analysis}"
                    )

                elif is_industry_question(user_msg):
                    industry_response = get_industry_chat_response(user_msg)
                    reply_line(
                        reply_token,
                        f"🌎 WengStock 產業觀察：\n\n{industry_response}"
                    )

                else:
                    chat_response = get_stock_chat_response(user_msg)
                    reply_line(
                        reply_token,
                        f"🤖 WengStock AI：\n\n{chat_response}"
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