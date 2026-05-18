"""
WengStock AI — 排程系統
台美股雙軌制五大定時任務 + FastAPI lifespan 生命週期管理
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import AsyncGenerator

import pytz
import httpx
import yfinance as yf

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

# ─────────────────────────────────────────────
# Logging 設定
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("wengstock.scheduler")


# ─────────────────────────────────────────────
# 時區工具（調試 / 日誌用）
# ─────────────────────────────────────────────

_NY_TZ = pytz.timezone("America/New_York")
_TW_TZ = pytz.timezone("Asia/Taipei")


def get_market_tz_info() -> dict:
    """
    回傳當前美東時間與 DST 狀態，方便日誌確認觸發時間是否正確。
    DST 期間（3月~11月）：ET = UTC-4，台北 +12H
    非 DST（11月~3月）  ：ET = UTC-5，台北 +13H
    """
    now_ny = datetime.now(_NY_TZ)
    now_tw = datetime.now(_TW_TZ)
    is_dst = bool(now_ny.dst())
    offset_hours = int(now_ny.utcoffset().total_seconds() / 3600)

    return {
        "ny_time":      now_ny.strftime("%Y-%m-%d %H:%M %Z"),
        "tw_time":      now_tw.strftime("%Y-%m-%d %H:%M %Z"),
        "is_dst":       is_dst,
        "utc_offset":   f"UTC{offset_hours:+d}",
        "market_open_tw": "21:30 台北" if is_dst else "22:30 台北",
        "market_close_tw": "04:00 台北" if is_dst else "05:00 台北",
    }


# ─────────────────────────────────────────────
# 全域限流器與行銷資料暫存
# ─────────────────────────────────────────────

_OPENAI_SEMAPHORE = asyncio.Semaphore(5)

# 今日 High Quality setups 暫存（供行銷任務挑選最高 RR）
_today_high_quality_setups: list[dict] = []


def record_high_quality_setup(setup: dict) -> None:
    """
    由主業務邏輯呼叫（分析完成後），把 High Quality 的 setup 登記進來。
    setup 至少要含：symbol, rr_ratio, rating, setup_type, market
    """
    if setup.get("rating", "").lower().startswith("high"):
        _today_high_quality_setups.append(setup)


def _reset_daily_setups() -> None:
    """每日午夜清除昨日 High Quality setup，避免跨日累積。"""
    _today_high_quality_setups.clear()
    log.info("🔄 今日 High Quality setup 清單已重置")


# ─────────────────────────────────────────────
# 警報掃描（每 30 分鐘）
# ─────────────────────────────────────────────

async def _task_check_alerts() -> None:
    """
    掃描所有活躍的條件式價格警報。
    觸發後立即 push LINE 通知，並在 alerts.json 標記已觸發。
    """
    import os
    from alerts import get_all_active_symbols, scan_and_mark_triggered

    line_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
    symbols    = get_all_active_symbols()

    if not symbols:
        return

    log.info(f"🔔 [警報掃描] 掃描 {len(symbols)} 個股票...")

    # 批次抓最新收盤價（1分鐘線取最後一根）
    current_prices: dict[str, float] = {}
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(period="1d", interval="1m")
            if not hist.empty:
                current_prices[sym] = float(hist["Close"].iloc[-1])
        except Exception as e:
            log.warning(f"[警報掃描] {sym} 取價失敗: {e}")
        await asyncio.sleep(0.2)

    if not current_prices:
        return

    triggered = scan_and_mark_triggered(current_prices)

    for alert in triggered:
        user_id        = alert["user_id"]
        sym            = alert["symbol"]
        target         = alert["target_price"]
        actual         = alert.get("triggered_price", target)
        direction_text = "跌到" if alert.get("direction") == "below" else "漲到"

        msg = (
            f"🔔【價格警報觸發！】\n\n"
            f"{sym} 已{direction_text} {target:.2f}！\n"
            f"目前價格：{actual:.2f}\n\n"
            f"輸入 {sym} 立即取得最新 Swing 分析 📊\n"
            f"輸入 /exit {sym} 分析出場時機 📤"
        )
        await send_line_push(user_id, msg)
        log.info(f"🔔 [警報] {sym} 已推播 → {user_id[:8]}...")

    log.info(f"🔔 [警報掃描] 完成，觸發 {len(triggered)} 個")


# ─────────────────────────────────────────────
# 勝率自動結算（每日盤後）
# ─────────────────────────────────────────────

_TRADES_FILE        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.json")
_TIMEOUT_CAL_DAYS   = 30   # 超過 30 日曆天視為 timeout（未觸發進場條件）


async def _task_settle_outcomes() -> None:
    """
    每日盤後自動結算：
    - 對近 60 天內、尚無 outcome 的 setup，抓歷史 OHLC 比對
    - stop_loss_price 先被觸發 → loss
    - target_1 先被觸發     → win
    - 超過 30 日曆天都未觸發 → timeout
    結算結果寫回 trades.json，讓 /stats 能顯示真實勝率。
    """
    log.info("📊 [勝率結算] 開始")

    try:
        with open(_TRADES_FILE, "r", encoding="utf-8") as f:
            trades = json.load(f)
    except Exception as e:
        log.warning(f"[勝率結算] 讀取 trades.json 失敗: {e}")
        return

    updated = False

    for trade in trades:
        if trade.get("outcome"):
            continue

        try:
            setup_dt = datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue

        days_elapsed = (datetime.now() - setup_dt).days
        if days_elapsed < 2:
            continue

        symbol     = trade.get("symbol", "")
        stop_price = trade.get("stop_loss_price") or trade.get("stop_loss", 0)
        target     = trade.get("target_1", 0)

        if not symbol or not stop_price or not target:
            continue

        # Timeout（等超過 30 天都沒進場，視為 setup 失效）
        if days_elapsed > _TIMEOUT_CAL_DAYS:
            trade["outcome"]      = "timeout"
            trade["outcome_date"] = datetime.now().strftime("%Y-%m-%d")
            updated = True
            log.info(f"[勝率結算] {symbol} → timeout（{days_elapsed}日未觸發）")
            continue

        try:
            loop = asyncio.get_event_loop()
            hist = await loop.run_in_executor(
                None,
                lambda s=symbol, d=setup_dt: yf.Ticker(s).history(
                    start=d.strftime("%Y-%m-%d"), interval="1d", auto_adjust=False
                ),
            )

            if hist.empty:
                continue

            outcome = outcome_date = outcome_price = None

            for ts, row in hist.iterrows():
                low  = float(row["Low"])
                high = float(row["High"])

                if low <= stop_price:
                    outcome, outcome_date, outcome_price = "loss", str(ts.date()), stop_price
                    break
                if high >= target:
                    outcome, outcome_date, outcome_price = "win",  str(ts.date()), target
                    break

            if outcome:
                trade["outcome"]       = outcome
                trade["outcome_date"]  = outcome_date
                trade["outcome_price"] = outcome_price
                updated = True
                log.info(f"[勝率結算] {symbol} → {outcome} @ {outcome_date}")

        except Exception as e:
            log.warning(f"[勝率結算] {symbol} 結算失敗: {e}")

    if updated:
        with open(_TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=4, ensure_ascii=False)

    log.info("📊 [勝率結算] 完成")


# ─────────────────────────────────────────────
# Mock 接口（後續直接替換成真實實作）
# ─────────────────────────────────────────────

async def send_line_push(line_user_id: str, content: str | dict) -> None:
    """
    推播 LINE 訊息給指定用戶。
    content 為 str 時發純文字；為 dict 時視為 Flex Message payload。
    【待替換】串接真實 LINE Push Message API。
    """
    log.info(f"[LINE PUSH] → {line_user_id[:8]}... | {str(content)[:80]}...")


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
async def ai_generate_report(model: str, prompt: str) -> str:
    """
    呼叫 OpenAI 生成報告文字，含指數退避重試（RateLimit / Timeout / Connection）。
    並發上限由 _OPENAI_SEMAPHORE（5）控制，避免批次觸發 429。
    """
    async with _OPENAI_SEMAPHORE:
        log.info(f"[AI] model={model} | prompt={prompt[:60]}...")
        # 實際串接時：
        # import os
        # client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # resp = await client.chat.completions.create(
        #     model=model,
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=600,
        # )
        # return resp.choices[0].message.content
        return f"[Mock Report @ {datetime.now().strftime('%H:%M:%S')}] {prompt[:40]}..."


# ─────────────────────────────────────────────
# 資料庫查詢（Mock，後續替換 SQLAlchemy Session）
# ─────────────────────────────────────────────

def db_get_watchlist(market_type: str) -> list[dict]:
    """
    模擬從 PostgreSQL 撈取自選股清單。
    實際 SQL：
        SELECT u.line_user_id, w.stock_code
        FROM watchlist w JOIN users u ON w.user_id = u.id
        WHERE w.market_type = :market_type
    """
    _mock_data = {
        "TW": [
            {"line_user_id": "U_alice", "stock_code": "2330"},
            {"line_user_id": "U_bob",   "stock_code": "2454"},
            {"line_user_id": "U_alice", "stock_code": "2317"},
        ],
        "US": [
            {"line_user_id": "U_alice", "stock_code": "NVDA"},
            {"line_user_id": "U_bob",   "stock_code": "TSLA"},
            {"line_user_id": "U_carol", "stock_code": "AMD"},
        ],
    }
    return _mock_data.get(market_type, [])


def db_get_all_us_watchlist_symbols() -> list[str]:
    """撈取所有用戶追蹤的美股代號（去重）"""
    rows = db_get_watchlist("US")
    return list({r["stock_code"] for r in rows})


def db_get_users_tracking(stock_code: str) -> list[str]:
    """找出追蹤特定股票的所有用戶 line_user_id"""
    rows = db_get_watchlist("US")
    return [r["line_user_id"] for r in rows if r["stock_code"] == stock_code]


# ─────────────────────────────────────────────
# 財報日曆（yfinance 實作）
# ─────────────────────────────────────────────

def get_earnings_in_next_24h(symbols: list[str]) -> list[dict]:
    """
    檢查哪些股票在未來 24 小時內發布財報。
    回傳 [{"symbol": "NVDA", "earnings_date": datetime}, ...]
    """
    upcoming: list[dict] = []
    now  = datetime.utcnow()
    end  = now + timedelta(hours=24)

    for sym in symbols:
        try:
            cal = yf.Ticker(sym).calendar
            if cal is None or cal.empty:
                continue

            # yfinance calendar 的欄位因版本而異
            date_col = next(
                (c for c in cal.columns if "Earnings" in c),
                None,
            )
            if not date_col:
                continue

            for raw_date in cal[date_col].dropna():
                dt = (
                    raw_date.to_pydatetime()
                    if hasattr(raw_date, "to_pydatetime")
                    else datetime.combine(raw_date, datetime.min.time())
                )
                if now <= dt <= end:
                    upcoming.append({"symbol": sym, "earnings_date": dt})
                    break

        except Exception as e:
            log.warning(f"財報日曆查詢失敗 [{sym}]: {e}")

    return upcoming


# ─────────────────────────────────────────────
# 任務一：台股盤前簡報（週一~五 08:30）
# ─────────────────────────────────────────────

async def _task_tw_premarket() -> None:
    log.info("📋 [任務一] 台股盤前簡報 — 開始")
    rows = db_get_watchlist("TW")

    # 依用戶分組
    by_user: dict[str, list[str]] = {}
    for r in rows:
        by_user.setdefault(r["line_user_id"], []).append(r["stock_code"])

    for user_id, stocks in by_user.items():
        try:
            prompt = (
                f"今日台股大盤展望與以下自選股盤前分析（試撮行情）：\n"
                f"股票：{', '.join(stocks)}\n"
                "請提供：1.大盤方向 2.各股試撮強弱 3.今日操作建議。"
                "語氣簡潔，適合手機閱讀。"
            )
            report = await ai_generate_report("gpt-4o", prompt)
            await send_line_push(user_id, f"🌅 台股盤前簡報\n\n{report}")

        except Exception as e:
            log.error(f"台股盤前簡報失敗 [{user_id}]: {e}")

        await asyncio.sleep(0.3)

    log.info(f"📋 [任務一] 完成，推播 {len(by_user)} 位用戶")


# ─────────────────────────────────────────────
# 任務二：台股收盤籌碼戰報（週一~五 17:00）
# ─────────────────────────────────────────────

async def _task_tw_close() -> None:
    log.info("🌆 [任務二] 台股收盤籌碼戰報 — 開始")
    rows = db_get_watchlist("TW")
    by_user: dict[str, list[str]] = {}
    for r in rows:
        by_user.setdefault(r["line_user_id"], []).append(r["stock_code"])

    for user_id, stocks in by_user.items():
        for sym in stocks:
            try:
                yf_sym = f"{sym}.TW"
                ticker = yf.Ticker(yf_sym)
                hist   = ticker.history(period="60d", interval="1d", auto_adjust=False)

                if hist.empty:
                    log.warning(f"台股資料為空：{yf_sym}")
                    continue

                latest = hist.iloc[-1]

                # 簡易指標（正式版請接 add_indicators）
                close  = round(float(latest["Close"]), 2)
                volume = int(latest["Volume"])

                prompt = (
                    f"台股 {sym} 收盤分析：\n"
                    f"收盤價：{close}｜成交量：{volume:,}\n"
                    "請分析：1.今日K線含義 2.主力籌碼動向 3.明日操作建議。"
                )
                report = await ai_generate_report("gpt-4o", prompt)
                await send_line_push(user_id, f"🌆 {sym} 收盤戰報\n\n{report}")

            except Exception as e:
                log.error(f"台股收盤戰報失敗 [{sym} / {user_id}]: {e}")

            await asyncio.sleep(0.3)

    log.info(f"🌆 [任務二] 完成")


# ─────────────────────────────────────────────
# 任務三：美股財報日曆主動防禦（每日 20:00）
# ─────────────────────────────────────────────

async def _task_earnings_defense() -> None:
    log.info("📅 [任務三] 美股財報風控防禦 — 開始")
    tracked_symbols = db_get_all_us_watchlist_symbols()
    upcoming        = get_earnings_in_next_24h(tracked_symbols)

    if not upcoming:
        log.info("📅 [任務三] 未來 24H 無追蹤股票財報，跳過")
        return

    for item in upcoming:
        sym          = item["symbol"]
        earn_dt      = item["earnings_date"]
        user_ids     = db_get_users_tracking(sym)

        for user_id in user_ids:
            try:
                prompt = (
                    f"⚠️ 財報風控提醒：{sym} 將於 {earn_dt.strftime('%m/%d %H:%M')} UTC 發布財報。\n"
                    "請生成：1.財報前市場預期 2.持倉風險評估 3.建議動作（持有/減倉/觀望）。"
                    "語氣直接，重點在風控，不鼓勵賭博式持倉。"
                )
                report = await ai_generate_report("gpt-4o-mini", prompt)
                await send_line_push(
                    user_id,
                    f"📅 財報風控提醒 — {sym}\n"
                    f"財報時間：{earn_dt.strftime('%m/%d %H:%M')} UTC\n\n"
                    f"{report}",
                )
                log.info(f"財報提醒已推播 [{sym} → {user_id[:8]}...]")

            except Exception as e:
                log.error(f"財報防禦推播失敗 [{sym} / {user_id}]: {e}")

    log.info(f"📅 [任務三] 完成，共 {len(upcoming)} 檔財報提醒")


# ─────────────────────────────────────────────
# 任務四：美股盤前簡報（週一~五 20:30 夏令）
# ─────────────────────────────────────────────

async def _task_us_premarket() -> None:
    log.info("🌌 [任務四] 美股盤前簡報 — 開始")
    rows = db_get_watchlist("US")
    by_user: dict[str, list[str]] = {}
    for r in rows:
        by_user.setdefault(r["line_user_id"], []).append(r["stock_code"])

    for user_id, stocks in by_user.items():
        try:
            prompt = (
                f"美股開盤前 30 分鐘策略簡報。\n"
                f"追蹤股票：{', '.join(stocks)}\n"
                "請提供：1.今日大盤風險（QQQ/SPY方向）2.各股盤前動能 3.今晚操作策略。"
                "如有重要總經數據（CPI/FED/PPI）請納入分析。"
            )
            report = await ai_generate_report("gpt-4o", prompt)
            await send_line_push(user_id, f"🌌 美股盤前簡報\n\n{report}")

        except Exception as e:
            log.error(f"美股盤前簡報失敗 [{user_id}]: {e}")

        await asyncio.sleep(0.3)

    log.info(f"🌌 [任務四] 完成，推播 {len(by_user)} 位用戶")


# ─────────────────────────────────────────────
# 任務五：美股收盤戰報（週二~六 06:00）
# ─────────────────────────────────────────────

async def _task_us_close() -> None:
    log.info("🌅 [任務五] 美股收盤戰報 — 開始")
    rows = db_get_watchlist("US")
    by_user: dict[str, list[str]] = {}
    for r in rows:
        by_user.setdefault(r["line_user_id"], []).append(r["stock_code"])

    for user_id, stocks in by_user.items():
        for sym in stocks:
            try:
                ticker = yf.Ticker(sym)
                hist   = ticker.history(period="60d", interval="1d", auto_adjust=False)

                if hist.empty:
                    log.warning(f"美股資料為空：{sym}")
                    continue

                latest  = hist.iloc[-1]
                close   = round(float(latest["Close"]), 2)
                volume  = int(latest["Volume"])

                # ATR 動態止損計算（簡版）
                atr_14  = (hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1]
                atr_stop = round(close - float(atr_14) * 1.5, 2)

                prompt = (
                    f"{sym} 美股收盤戰報：\n"
                    f"收盤價：${close:,.2f}｜成交量：{volume:,}｜ATR 止損參考：${atr_stop:,.2f}\n"
                    "請分析：1.今日K線收型 2.是否觸發ATR止損 3.明日盤前關注重點。"
                    "語氣直接，風控優先。"
                )
                report = await ai_generate_report("gpt-4o", prompt)
                await send_line_push(
                    user_id,
                    f"🌅 {sym} 美股收盤戰報\n"
                    f"收盤：${close:,.2f} | ATR 止損：${atr_stop:,.2f}\n\n"
                    f"{report}",
                )

            except Exception as e:
                log.error(f"美股收盤戰報失敗 [{sym} / {user_id}]: {e}")

            await asyncio.sleep(0.3)

    log.info("🌅 [任務五] 完成")


# ─────────────────────────────────────────────
# 行銷自動化（任務六 / 七）：High Quality Setup → 自媒體短文
# ─────────────────────────────────────────────

async def _post_to_webhook(payload: dict) -> None:
    """
    預留接口：將行銷文字 POST 到外部 Webhook（如 n8n / Zapier / Threads API）。
    未來串接時替換 MARKETING_WEBHOOK_URL 即可，無需改動業務邏輯。
    payload 格式：{"market": "TW"|"US", "symbol": str, "post_text": str, "rr_ratio": float}
    """
    import os
    webhook_url = os.getenv("MARKETING_WEBHOOK_URL", "")
    if not webhook_url:
        log.info(f"[行銷] Webhook URL 未設定，輸出至日誌：\n{payload.get('post_text', '')}")
        return

    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.post(webhook_url, json=payload)
            if resp.status_code == 200:
                log.info(f"[行銷] Webhook 推送成功 → {payload.get('symbol')}")
            else:
                log.warning(f"[行銷] Webhook 回應 {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        log.error(f"[行銷] Webhook 推送失敗: {e}")


async def _task_marketing_post(market: str) -> None:
    """
    從今日 High Quality setups 中挑選風報比最高的一檔，
    用 gpt-4o-mini 生成適合 Threads / 自媒體的行銷短文，
    並呼叫 _post_to_webhook() 送出。
    market: "TW" | "US"
    """
    log.info(f"📣 [行銷] {market} 行銷發文任務 — 開始")

    # 篩選出今日指定市場的 High Quality setups
    candidates = [
        s for s in _today_high_quality_setups
        if s.get("market", "").upper() == market
    ]

    if not candidates:
        log.info(f"📣 [行銷] 今日無 {market} High Quality setup，跳過發文")
        return

    # 挑選風報比最高的一檔
    best = max(candidates, key=lambda s: float(s.get("rr_ratio", 0)))
    sym       = best.get("symbol", "N/A")
    rr        = float(best.get("rr_ratio", 0))
    setup_type = best.get("setup_type", "N/A")
    price     = best.get("price", 0)
    target1   = best.get("target_1", 0)
    stop_loss = best.get("stop_loss", 0)

    log.info(f"📣 [行銷] 挑選最佳 setup：{sym}（{market}）RR={rr:.1f}R")

    prompt = f"""
你是一位擅長用自媒體語氣分享股票 Swing Trading 機會的台灣財經 KOL。

今日系統偵測到一個高品質波段機會：
- 股票：{sym}（{market}市場）
- 類型：{setup_type}
- 現價：{price:.2f}
- 目標：{target1:.2f}
- 止損：{stop_loss:.2f}
- 風報比：{rr:.1f}R

請用繁體中文，以 Threads / Instagram 短文風格，寫一篇 150 字以內的行銷短文。
要求：
1. 語氣輕鬆、有吸引力，帶出「AI 選股」的科技感
2. 點出風報比亮點（{rr:.1f}R）
3. 結尾加上免責聲明（非投資建議）
4. 不誇大保證獲利，保持合規
5. 加 2~3 個相關 hashtag
""".strip()

    try:
        post_text = await ai_generate_report("gpt-4o-mini", prompt)
        log.info(f"📣 [行銷] 短文生成完成 ({len(post_text)} chars)")

        payload = {
            "market":    market,
            "symbol":    sym,
            "rr_ratio":  rr,
            "post_text": post_text,
        }
        await _post_to_webhook(payload)

    except Exception as e:
        log.error(f"📣 [行銷] 發文失敗 [{sym}]: {e}")

    log.info(f"📣 [行銷] {market} 行銷發文任務 — 完成")


# ─────────────────────────────────────────────
# APScheduler 同步包裝（Bridge sync → async）
# ─────────────────────────────────────────────

def _run_async(coro) -> None:
    """讓 APScheduler（同步）觸發 async 任務的橋接函數。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(coro)
        else:
            loop.run_until_complete(coro)
    except RuntimeError:
        # uvicorn thread-pool threads have no current event loop — create one
        asyncio.run(coro)
    except Exception as e:
        log.error(f"排程橋接失敗: {e}")


# ─────────────────────────────────────────────
# 排程器建構
# ─────────────────────────────────────────────

def build_scheduler() -> BackgroundScheduler:
    """
    台股任務  → Asia/Taipei 時區（台股沒有 DST，寫死安全）
    美股任務  → America/New_York 時區（APScheduler 自動處理 DST 切換）

    美東觸發時間對照：
    ┌─────────────────────┬───────────┬───────────┐
    │ 任務                │ 夏令(ET)  │ 冬令(ET)  │
    ├─────────────────────┼───────────┼───────────┤
    │ 財報防禦            │ 07:00 ET  │ 07:00 ET  │  ← 美股開盤前 2.5H
    │ 美股盤前簡報        │ 08:30 ET  │ 08:30 ET  │  ← 美股開盤前 1H
    │ 美股收盤戰報        │ 16:30 ET  │ 16:30 ET  │  ← 美股收盤後 30min
    └─────────────────────┴───────────┴───────────┘
    對應台北時間（僅供參考，不寫進程式碼）：
    ┌─────────────────────┬───────────┬───────────┐
    │ 任務                │ 夏令(TW)  │ 冬令(TW)  │
    ├─────────────────────┼───────────┼───────────┤
    │ 財報防禦            │ 19:00 TW  │ 20:00 TW  │
    │ 美股盤前簡報        │ 20:30 TW  │ 21:30 TW  │
    │ 美股收盤戰報        │ 04:30 TW  │ 05:30 TW  │
    └─────────────────────┴───────────┴───────────┘
    """
    tz_info = get_market_tz_info()
    log.info(
        f"排程器初始化 | 紐約={tz_info['ny_time']} | "
        f"DST={'開啟' if tz_info['is_dst'] else '關閉'} | "
        f"{tz_info['utc_offset']}"
    )

    sch = BackgroundScheduler(timezone="Asia/Taipei")

    # ── 台股任務（Asia/Taipei，無 DST 問題）─────────────────

    # 任務一：台股盤前簡報（週一~五 08:30 台北）
    sch.add_job(
        lambda: _run_async(_task_tw_premarket()),
        CronTrigger(day_of_week="mon-fri", hour=8, minute=30,
                    timezone="Asia/Taipei"),
        id="tw_premarket",
        name="🌅 台股盤前簡報",
        replace_existing=True,
    )

    # 任務二：台股收盤籌碼戰報（週一~五 17:00 台北）
    sch.add_job(
        lambda: _run_async(_task_tw_close()),
        CronTrigger(day_of_week="mon-fri", hour=17, minute=0,
                    timezone="Asia/Taipei"),
        id="tw_close",
        name="🌆 台股收盤戰報",
        replace_existing=True,
    )

    # ── 美股任務（America/New_York，自動 DST）────────────────

    # 任務三：財報防禦（每日 07:00 ET = 開盤前 2.5H）
    # 比較重要的財報通常在盤前或盤後，07:00 ET 能提前預警
    sch.add_job(
        lambda: _run_async(_task_earnings_defense()),
        CronTrigger(hour=7, minute=0,
                    timezone="America/New_York"),
        id="earnings_defense",
        name="📅 美股財報風控防禦",
        replace_existing=True,
    )

    # 任務四：美股盤前簡報（週一~五 08:30 ET = 開盤前 1H）
    sch.add_job(
        lambda: _run_async(_task_us_premarket()),
        CronTrigger(day_of_week="mon-fri", hour=8, minute=30,
                    timezone="America/New_York"),
        id="us_premarket",
        name="🌌 美股盤前簡報",
        replace_existing=True,
    )

    # 任務五：美股收盤戰報（週一~五 16:30 ET = 收盤後 30min）
    # day_of_week="mon-fri"：ET 的週一~五 → 台北對應週二~六，無需手動換算
    sch.add_job(
        lambda: _run_async(_task_us_close()),
        CronTrigger(day_of_week="mon-fri", hour=16, minute=30,
                    timezone="America/New_York"),
        id="us_close",
        name="🌅 美股收盤戰報",
        replace_existing=True,
    )

    # ── 每日重置（午夜台北時間）────────────────────────────────
    sch.add_job(
        _reset_daily_setups,
        CronTrigger(hour=0, minute=0, timezone="Asia/Taipei"),
        id="daily_reset",
        name="🔄 每日 Setup 清單重置",
        replace_existing=True,
    )

    # ── 警報掃描（每 30 分鐘，全天候）─────────────────────────
    sch.add_job(
        lambda: _run_async(_task_check_alerts()),
        CronTrigger(minute="*/30"),
        id="alert_scan",
        name="🔔 條件式價格警報掃描",
        replace_existing=True,
    )

    # ── 勝率結算：台股盤後（週一~五 18:00 台北）────────────────
    sch.add_job(
        lambda: _run_async(_task_settle_outcomes()),
        CronTrigger(day_of_week="mon-fri", hour=18, minute=0,
                    timezone="Asia/Taipei"),
        id="settle_tw",
        name="📊 台股勝率結算",
        replace_existing=True,
    )

    # ── 勝率結算：美股盤後（週一~五 17:30 ET）──────────────────
    sch.add_job(
        lambda: _run_async(_task_settle_outcomes()),
        CronTrigger(day_of_week="mon-fri", hour=17, minute=30,
                    timezone="America/New_York"),
        id="settle_us",
        name="📊 美股勝率結算",
        replace_existing=True,
    )

    # ── 行銷自動化任務（收盤後觸發）────────────────────────────

    # 任務六：台股行銷發文（週一~五 17:30 台北，收盤戰報後 30min）
    sch.add_job(
        lambda: _run_async(_task_marketing_post("TW")),
        CronTrigger(day_of_week="mon-fri", hour=17, minute=30,
                    timezone="Asia/Taipei"),
        id="marketing_tw",
        name="📣 台股行銷發文",
        replace_existing=True,
    )

    # 任務七：美股行銷發文（週一~五 17:00 ET，收盤戰報後 30min）
    sch.add_job(
        lambda: _run_async(_task_marketing_post("US")),
        CronTrigger(day_of_week="mon-fri", hour=17, minute=0,
                    timezone="America/New_York"),
        id="marketing_us",
        name="📣 美股行銷發文",
        replace_existing=True,
    )

    return sch


# ─────────────────────────────────────────────
# FastAPI Lifespan（取代舊的 @on_event）
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI 最新 lifespan 模式。
    startup：啟動排程器、建立資料表。
    shutdown：安全停止排程器，防止記憶體洩漏。
    """
    # ── startup ──
    log.info("🚀 WengStock AI 啟動中...")

    # 建立資料表（首次啟動時）
    try:
        from models import create_tables
        create_tables()
        log.info("✅ 資料表確認完成")
    except Exception as e:
        log.warning(f"資料表建立跳過（可能尚未設定 DB）: {e}")

    scheduler = build_scheduler()
    scheduler.start()

    _log_registered_jobs(scheduler)
    app.state.scheduler = scheduler   # 掛到 app.state，方便路由層存取

    log.info("✅ 排程器啟動完成")

    yield   # ← 伺服器運行期間在這裡

    # ── shutdown ──
    log.info("🛑 WengStock AI 關閉中，停止排程器...")
    scheduler.shutdown(wait=False)
    log.info("✅ 排程器已安全停止")


def _log_registered_jobs(sch: BackgroundScheduler) -> None:
    log.info("📋 已註冊定時任務：")
    for job in sch.get_jobs():
        log.info(f"  ├─ [{job.id}] {job.name}  next={job.next_run_time}")


# ─────────────────────────────────────────────
# 掛載到 FastAPI App（在 main.py 使用）
# ─────────────────────────────────────────────
#
# from scheduler import lifespan
# app = FastAPI(lifespan=lifespan)
#
