import os
import re
import json
import asyncio

from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import httpx
from openai import AsyncOpenAI

# 從現有 agent.py 引用核心業務邏輯（不重寫）
from agent import (
    add_indicators,
    detect_swing_setup,
    get_market_filter,
    get_stock_news,
    get_earnings_warning,
    is_stock_related,
    is_direct_ticker,
    is_industry_question,
    log_trade,
    analyze_trade_history,
    calc_position_size,
    format_position_size,
    scan_market_sync,
)
from alerts import add_alert, remove_alert, format_user_alerts
from watchlist import (
    add_watch, remove_watch, format_user_watchlist,
    is_new_user, mark_user_seen,
)
from gorilla.screener import screen_us, screen_tw, run_daily_scan
from gorilla.position_manager import (
    record_entry, close_position, format_positions,
    check_positions_sync, get_all_user_ids,
)
from gorilla.gorilla_flex import build_gorilla_flex
from gorilla.market_filter import format_market_status
from gorilla.subscribe import subscribe as gorilla_subscribe, unsubscribe as gorilla_unsubscribe
from market_router import MarketRouter
from models import watchlist_router, create_tables
from flex_builder import build_stock_report_flex
from scheduler import lifespan, get_market_tz_info, record_high_quality_setup

# ─────────────────────────────────────────────
# 環境設定
# ─────────────────────────────────────────────

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path)

LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=OPENAI_KEY)
router = MarketRouter()

app = FastAPI(title="WengStock AI", version="2.0.0", lifespan=lifespan)
app.include_router(watchlist_router)


# ─────────────────────────────────────────────
# LINE API helpers（非同步）
# ─────────────────────────────────────────────

async def reply_line(reply_token: str, text: str) -> None:
    """立刻回覆（Reply Token，30 秒內有效）"""
    async with httpx.AsyncClient(timeout=10) as http:
        await http.post(
            "https://api.line.me/v2/bot/message/reply",
            headers={"Authorization": f"Bearer {LINE_TOKEN}"},
            json={
                "replyToken": reply_token,
                "messages": [{"type": "text", "text": text[:4900]}],
            },
        )


async def push_flex(user_id: str, flex_payload: dict) -> None:
    """推送 Flex Message（卡片樣式）"""
    async with httpx.AsyncClient(timeout=10) as http:
        resp = await http.post(
            "https://api.line.me/v2/bot/message/push",
            headers={"Authorization": f"Bearer {LINE_TOKEN}"},
            json={"to": user_id, "messages": [flex_payload]},
        )
        if resp.status_code != 200:
            print(f"❌ Flex Push 失敗 [{resp.status_code}]: {resp.text}")


async def push_line(user_id: str, text: str) -> None:
    """背景任務完成後推送（不受 Reply Token 限制）"""
    async with httpx.AsyncClient(timeout=10) as http:
        resp = await http.post(
            "https://api.line.me/v2/bot/message/push",
            headers={"Authorization": f"Bearer {LINE_TOKEN}"},
            json={
                "to": user_id,
                "messages": [{"type": "text", "text": text[:4900]}],
            },
        )
        if resp.status_code != 200:
            print(f"❌ Push 失敗 [{resp.status_code}]: {resp.text}")


# ─────────────────────────────────────────────
# 核心分析（背景任務，使用 gpt-4o）
# ─────────────────────────────────────────────

async def run_swing_analysis(symbol: str, user_id: str) -> None:
    """
    在背景執行完整的 Swing 分析，完成後 Push 報告給用戶。
    使用 gpt-4o 做深度分析。
    """
    try:
        # 抓資料（同步 yfinance 在 executor 裡跑，避免 block event loop）
        loop = asyncio.get_event_loop()
        fetch_result = await loop.run_in_executor(None, router.route, symbol)

        if fetch_result.daily.empty or fetch_result.h4.empty or fetch_result.h1.empty:
            await push_line(user_id, f"❌ 找不到 {symbol} 的足夠資料，請確認代號是否正確。")
            return

        daily = add_indicators(fetch_result.daily).dropna()
        h4 = add_indicators(fetch_result.h4).dropna()
        h1 = add_indicators(fetch_result.h1).dropna()

        if len(daily) < 50:
            await push_line(user_id, f"❌ {symbol} 歷史資料不足，無法做 Swing 判斷。")
            return

        market = await loop.run_in_executor(None, get_market_filter)
        setup = detect_swing_setup(daily, h4, h1, market)
        news = await loop.run_in_executor(None, get_stock_news, fetch_result.symbol)
        log_trade(fetch_result.symbol, setup)
        # 行銷自動化：登記今日 High Quality setup
        record_high_quality_setup({
            **setup,
            "symbol": fetch_result.symbol,
            "market": fetch_result.market.value.replace("_MARKET", ""),  # "TW" | "US"
        })

        news_text = "\n".join([f"- {n['title']}" for n in news[:5]])

        # gpt-4o 深度分析
        prompt = _build_analysis_prompt(fetch_result.symbol, setup, news_text)
        response = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0.35,
            max_tokens=800,
            messages=[
                {"role": "system", "content": "你是專業、冷靜、重視風險的 Swing Trading 分析助理。"},
                {"role": "user", "content": prompt},
            ],
        )
        report_text = response.choices[0].message.content

        # 組裝 Flex Message（Bloomberg 暗色終端機風格）
        report_json = {
            **setup,
            "symbol":       fetch_result.symbol,
            "news_summary": "\n".join([n["title"] for n in news[:3]]),
        }
        flex_msg = build_stock_report_flex(report_json)
        await push_flex(user_id, flex_msg)
        await push_line(user_id, f"🤖 WengStock AI 分析：\n\n{report_text}")

        # ── No Trade 時，給出回踩目標與一鍵設警報
        rating = setup.get("rating", "").lower()
        if "no" in rating or "🔴" in setup.get("rating", ""):
            sym        = fetch_result.symbol
            entry_low  = setup.get("entry_zone_low", 0)
            entry_high = setup.get("entry_zone_high", 0)
            planned    = setup.get("planned_entry", entry_low)
            overheat   = setup.get("overheat_alert", False)
            storm      = setup.get("storm_mode", False)
            dev_pct    = setup.get("deviation_from_ma20_pct", 0)

            if storm:
                tip = (
                    f"⛈️ {sym} 現在大盤是風暴模式，整體市場不宜進場。\n"
                    "等大盤重新站上 MA200 再考慮。"
                )
            elif overheat:
                tip = (
                    f"🔥 {sym} 目前超漲 {dev_pct:.1f}%，需要等回踩。\n\n"
                    f"📍 等待區：{entry_low:.2f} – {entry_high:.2f}\n"
                    f"🎯 計畫進場：{planned:.2f}\n\n"
                    f"⚡ 一鍵設警報（到價通知你）：\n"
                    f"/alert {sym} {planned:.2f}"
                )
            else:
                tip = (
                    f"📍 {sym} 目前不符合進場條件。\n"
                    f"等待回踩至 {entry_low:.2f} – {entry_high:.2f} 再觀察。\n\n"
                    f"⚡ 設警報：/alert {sym} {planned:.2f}"
                )
            await push_line(user_id, tip)

    except asyncio.TimeoutError:
        await push_line(user_id, "⚠️ 分析超時，請稍後再試。")
    except Exception as e:
        err = str(e).lower()
        if "rate" in err or "429" in err:
            await push_line(user_id, f"⚠️ 資料來源暫時忙碌，請 30 秒後再試一次。")
        else:
            print(f"❌ 背景分析失敗 [{symbol}]: {e}")
            await push_line(user_id, f"⚠️ 分析時發生錯誤，請稍後再試。")


async def run_size_analysis(symbol: str, account_size: float, user_id: str) -> None:
    """
    計算並推播部位大小建議。
    先抓 setup，再呼叫 calc_position_size()，格式化後 push 給用戶。
    """
    try:
        loop         = asyncio.get_event_loop()
        fetch_result = await loop.run_in_executor(None, router.route, symbol)

        if fetch_result.daily.empty:
            await push_line(user_id, f"❌ 找不到 {symbol} 的資料。")
            return

        daily  = add_indicators(fetch_result.daily).dropna()
        h4     = add_indicators(fetch_result.h4).dropna()
        h1     = add_indicators(fetch_result.h1).dropna()
        market = await loop.run_in_executor(None, get_market_filter)
        setup  = detect_swing_setup(daily, h4, h1, market)

        if setup.get("rating", "").startswith("🔴"):
            await push_line(
                user_id,
                f"⚠️ {fetch_result.symbol} 目前評級為 {setup['rating']}，"
                "系統不建議現在進場，部位計算暫不執行。\n\n"
                f"等待區：{setup['entry_zone_low']:.2f} – {setup['entry_zone_high']:.2f}\n"
                "等回踩再算！",
            )
            return

        size   = calc_position_size(setup, account_size)
        result = format_position_size(fetch_result.symbol, size)
        await push_line(user_id, result)

    except Exception as e:
        print(f"❌ 部位計算失敗 [{symbol}]: {e}")
        await push_line(user_id, "⚠️ 部位計算時發生錯誤，請稍後再試。")


async def run_alert_setup(symbol: str, target_price: float, user_id: str) -> None:
    """
    抓取現價後設定條件式警報，並 push 確認訊息給用戶。
    """
    try:
        loop         = asyncio.get_event_loop()
        fetch_result = await loop.run_in_executor(None, router.route, symbol)

        if fetch_result.daily.empty:
            await push_line(user_id, f"❌ 找不到 {symbol} 的資料，無法設定警報。")
            return

        daily        = add_indicators(fetch_result.daily).dropna()
        current_price = float(daily["Close"].iloc[-1])

        msg = add_alert(user_id, fetch_result.symbol, target_price, current_price)
        await push_line(user_id, msg)

    except Exception as e:
        print(f"❌ 警報設定失敗 [{symbol}]: {e}")
        await push_line(user_id, "⚠️ 警報設定時發生錯誤，請稍後再試。")


async def run_exit_analysis(symbol: str, user_id: str) -> None:
    """
    專門的持倉出場分析：Trailing Stop、分批停利、大戶倒貨警報。
    用戶輸入 /exit NVDA 時觸發，回傳聚焦出場建議的大白話報告。
    """
    try:
        loop = asyncio.get_event_loop()
        fetch_result = await loop.run_in_executor(None, router.route, symbol)

        if fetch_result.daily.empty:
            await push_line(user_id, f"❌ 找不到 {symbol} 的資料，請確認代號是否正確。")
            return

        daily = add_indicators(fetch_result.daily).dropna()
        h4    = add_indicators(fetch_result.h4).dropna()
        h1    = add_indicators(fetch_result.h1).dropna()

        if len(daily) < 20:
            await push_line(user_id, f"❌ {symbol} 資料不足，無法分析出場訊號。")
            return

        market = await loop.run_in_executor(None, get_market_filter)
        setup  = detect_swing_setup(daily, h4, h1, market)

        exit_action    = setup.get("exit_action", "hold")
        exit_message   = setup.get("exit_message", "")
        trailing_stop  = setup.get("trailing_stop", 0)
        stop_loss_price = setup.get("stop_loss_price", 0)
        dev_pct        = setup.get("deviation_from_ma20_pct", 0)
        dist_alert     = setup.get("distribution_alert", False)

        prompt = f"""
你是 WengStock AI，用最溫柔堅定的語氣，幫用戶判斷手上的 {fetch_result.symbol} 現在要不要賣。
你的任務是出場分析，不是進場分析。

技術數據：
現價：{setup["price"]:.2f}
動態防守價（Trailing Stop）：{trailing_stop:.2f}
絕對止損價：{stop_loss_price:.2f}
出場動作：{exit_action}
出場訊號：{exit_message}
正乖離率（MA20）：{dev_pct:.1f}%
RSI：{setup["rsi"]:.2f}
量比：{setup["vol_ratio"]:.2f}（> 3x 且收黑/長上影 = 大戶倒貨警報）
大盤狀態：{setup["market_status"]}
大戶倒貨警報：{dist_alert}

請用繁體中文，像對長輩朋友說話：
1. 第一段：直接說「要賣 / 不要賣 / 先賣一部分」的結論
2. 第二段：解釋 Trailing Stop 防線在哪，跌破要怎麼做
3. 第三段：如果有觸發出場訊號，用大白話說清楚怎麼操作
4. 最後一句：一句安撫或風控提醒

不超過 8 句。語氣像 LINE 聊天，不像分析報告。
"""
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=400,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 WengStock AI，最良心的持倉出場分析助理。"
                        "只做出場分析，語氣溫柔但堅定，像對長輩說話。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        reply = response.choices[0].message.content
        await push_line(user_id, f"📤 {fetch_result.symbol} 出場分析\n\n{reply}")

    except Exception as e:
        print(f"❌ 出場分析失敗 [{symbol}]: {e}")
        await push_line(user_id, "⚠️ 出場分析時發生錯誤，請稍後再試。")


async def _get_swing_data(symbol: str) -> dict:
    """抓取並計算 Swing Setup，回傳 setup dict（失敗回傳空 dict）。"""
    try:
        loop         = asyncio.get_event_loop()
        fetch_result = await loop.run_in_executor(None, router.route, symbol)
        if fetch_result.daily.empty:
            return {}
        daily  = add_indicators(fetch_result.daily).dropna()
        h4     = add_indicators(fetch_result.h4).dropna()
        h1     = add_indicators(fetch_result.h1).dropna()
        market = await loop.run_in_executor(None, get_market_filter)
        return detect_swing_setup(daily, h4, h1, market)
    except Exception as e:
        print(f"⚠️ Swing data fetch failed [{symbol}]: {e}")
        return {}


async def run_gorilla_diagnosis(symbol: str, user_id: str) -> None:
    """大猩猩即時診斷（單支股票），通過後自動疊加 Swing 雙確認。"""
    try:
        is_tw  = symbol.isdigit()
        result = await (screen_tw(symbol) if is_tw else screen_us(symbol))

        earn = result.get("earnings_warning")

        if result.get("pass") and not earn:
            # 疊加 Swing 確認
            swing = await _get_swing_data(symbol)
            rating = swing.get("rating", "")
            if "high quality" in rating.lower() or "🟢" in rating:
                result["swing_setup"] = swing
                sig = "DUAL_CONFIRM"
            else:
                sig = "BUY"
        elif result.get("pass") and earn:
            sig = "BUY_WARN"
        else:
            sig = "NO_PASS"

        flex = build_gorilla_flex(sig, result)
        await push_flex(user_id, flex)

        if result.get("pass"):
            passes = "\n".join(f"✅ {p}" for p in result.get("passes", []))
            if sig == "DUAL_CONFIRM":
                sw = result["swing_setup"]
                await push_line(user_id,
                    f"🔥 {symbol} 大猩猩 + Swing 雙策略確認！\n\n"
                    f"{passes}\n\n"
                    f"Swing 評級：{sw.get('rating', 'N/A')}\n"
                    f"風報比：{sw.get('rr_ratio', 0):.1f}R\n\n"
                    f"建議以現價 {result['price']:.2f} 試單 5%，\n"
                    f"停損設在 {result['price'] * 0.925:.2f}（-7.5%）\n\n"
                    f"記錄進場：/gentry {symbol} {result['price']:.2f}")
            elif earn:
                await push_line(user_id,
                    f"🦍 {symbol} 基本面通過，但 ⚠️ {earn} 財報即將公布！\n\n"
                    f"{passes}\n\n"
                    "建議：等財報後確認方向再進場，\n"
                    "不要在財報前追高。\n\n"
                    f"財報後若股價站穩，再用：\n/gentry {symbol} [進場價]")
            else:
                await push_line(user_id,
                    f"🦍 {symbol} 通過大猩猩篩選！\n\n{passes}\n\n"
                    f"建議以現價 {result['price']:.2f} 試單 5%，\n"
                    f"停損設在 {result['price'] * 0.925:.2f}（-7.5%）\n\n"
                    f"記錄進場：/gentry {symbol} {result['price']:.2f}")
        else:
            fails  = "\n".join(f"❌ {f}" for f in result.get("fails", []))
            passes = "\n".join(f"✅ {p}" for p in result.get("passes", []))
            await push_line(user_id,
                f"🦍 {symbol} 目前不符合大猩猩條件\n\n{fails}\n\n已通過：\n{passes}")
    except Exception as e:
        print(f"❌ Gorilla 診斷失敗 [{symbol}]: {e}")
        await push_line(user_id, "⚠️ 大猩猩診斷失敗，請稍後再試。")


async def run_gorilla_scan(market: str, user_id: str) -> None:
    """每日大猩猩掃描結果推播"""
    try:
        result = await run_daily_scan(market)
        flag   = "🇺🇸" if market == "US" else "🇹🇼"
        label  = "美股" if market == "US" else "台股"

        if not result["market_safe"]:
            await push_line(user_id,
                f"🚫 {flag} {label}大盤偏弱，大猩猩策略暫停買進\n\n"
                f"{result['market_detail']}\n\n"
                "等大盤重新站上均線再掃描。")
            return

        picks = result.get("picks", [])
        if not picks:
            await push_line(user_id,
                f"🦍 {flag} {label}大猩猩掃描完成\n\n"
                "今日 watchlist 中無完全符合條件的標的。\n"
                "大盤安全但個股條件未到，繼續等待。")
            return

        # 推播前三名（加入 Swing 雙確認）
        dual_names = []
        for pick in picks[:3]:
            if pick.get("earnings_warning"):
                sig = "BUY_WARN"
            else:
                swing = await _get_swing_data(pick["ticker"])
                rating = swing.get("rating", "")
                if "high quality" in rating.lower() or "🟢" in rating:
                    pick["swing_setup"] = swing
                    sig = "DUAL_CONFIRM"
                    dual_names.append(pick["ticker"])
                else:
                    sig = "BUY"
            flex = build_gorilla_flex(sig, pick)
            await push_flex(user_id, flex)

        warn_names = [p["ticker"] for p in picks if p.get("earnings_warning")]
        names = "、".join(p["ticker"] for p in picks)
        warn_note = f"\n⚠️ {' / '.join(warn_names)} 財報警告，建議財報後再進場" if warn_names else ""
        dual_note = f"\n🔥 {' / '.join(dual_names)} 雙策略確認，優先考慮！" if dual_names else ""
        await push_line(user_id,
            f"🦍 {flag} {label}今日大猩猩精選：{names}\n"
            f"共 {len(picks)} 支通過篩選，以上為前 3 名。{warn_note}{dual_note}\n\n"
            "輸入代號查看完整 Swing 分析（如：NVDA）")

    except Exception as e:
        print(f"❌ Gorilla scan 失敗: {e}")
        await push_line(user_id, "⚠️ 掃描時發生錯誤，請稍後再試。")


async def run_morning_brief(user_id: str) -> None:
    """
    根據用戶自選股生成個人化盤前簡報，附帶真實價格快照。
    沒有自選股則用預設美股清單。
    """
    try:
        from watchlist import get_user_watchlist
        stocks = get_user_watchlist(user_id)
        is_personal = bool(stocks)
        if not stocks:
            stocks = ["NVDA", "TSLA", "AAPL", "AMD", "META", "QQQ", "SPY"]

        # 抓各股最新價格快照
        loop = asyncio.get_event_loop()
        snapshots = []
        for sym in stocks:
            try:
                fr    = await loop.run_in_executor(None, router.route, sym)
                daily = add_indicators(fr.daily).dropna()
                if daily.empty:
                    continue
                d      = daily.iloc[-1]
                price  = float(d["Close"])
                ema20  = float(d.get("EMA20", 0))
                rsi    = float(d.get("RSI", 0))
                chg_pct = (price / float(daily.iloc[-2]["Close"]) - 1) * 100 if len(daily) > 1 else 0
                snapshots.append(
                    f"{sym}: ${price:.2f} ({chg_pct:+.1f}%) | EMA20={ema20:.2f} | RSI={rsi:.1f}"
                )
            except Exception:
                continue

        snapshot_text = "\n".join(snapshots) if snapshots else "（無法取得即時數據）"
        label = "個人自選股" if is_personal else "熱門美股"

        prompt = (
            f"盤前簡報 — {label}\n\n"
            f"【今日各股即時快照】\n{snapshot_text}\n\n"
            "請根據以上數據提供：\n"
            "1. 大盤方向研判（看 SPY/QQQ/加權指數）\n"
            "2. 各股今日重點（一行一支，指出機會或風險）\n"
            "3. 最值得關注的 1–2 支與進場條件\n"
            "4. 風險提示（RSI 過熱、距 EMA20 過遠等）\n\n"
            "語氣簡潔直接，重視風控，繁體中文，適合手機閱讀。"
        )

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            max_tokens=700,
            messages=[
                {"role": "system", "content": "你是 WengStock AI 盤前分析助理，根據即時數據給出專業判斷。"},
                {"role": "user", "content": prompt},
            ],
        )
        report = response.choices[0].message.content
        header = "📈 個人盤前簡報\n" if is_personal else "📈 盤前簡報（輸入 /watch NVDA 可個人化）\n"
        await push_line(user_id, f"{header}\n{report}")

    except Exception as e:
        print(f"❌ 盤前簡報失敗 [{user_id}]: {e}")
        await push_line(user_id, "⚠️ 盤前簡報生成失敗，請稍後再試。")


async def run_scan_analysis(market: str, user_id: str) -> None:
    """掃描 watchlist，回傳高勝率 Setup 清單。"""
    try:
        loop    = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, scan_market_sync, market)
        mkt_label = "美股" if market == "US" else "台股"

        if not results:
            msg = (
                f"🔍 {mkt_label} 掃描完成\n\n"
                "目前 watchlist 中無高勝率 Setup。\n"
                "大多數標的可能處於超漲或大盤偏弱狀態。\n\n"
                "💡 建議做法：\n"
                "  1. 對感興趣的個股設回踩警報\n"
                "     例：/alert NVDA 900\n"
                "  2. 等大盤回測 MA20 後再掃描\n"
                "  3. 現在抱現金休息也是一種策略 💰"
            )
        else:
            lines = [f"🔍 {mkt_label} 高勝率 Setup — 今日機會：\n"]
            for r in results:
                badge = "🟢" if r["rating"].lower().startswith("high") else "🟡"
                lines.append(
                    f"{badge} {r['symbol']}  {r['setup_type']}\n"
                    f"   進場 {r['planned_entry']:.2f} | 目標 {r['target_1']:.2f} | {r['rr_ratio']:.1f}R\n"
                )
            lines.append("👆 輸入股票代號查看完整分析（例：NVDA）")
            msg = "\n".join(lines)

        await push_line(user_id, msg)

    except Exception as e:
        print(f"❌ Scan 失敗: {e}")
        await push_line(user_id, "⚠️ 掃描時發生錯誤，請稍後再試。")


async def run_compare_analysis(sym1: str, sym2: str, user_id: str) -> None:
    """同時分析兩支股票，給出孰優孰劣的建議。"""
    try:
        loop = asyncio.get_event_loop()
        r1, r2 = await asyncio.gather(
            loop.run_in_executor(None, router.route, sym1),
            loop.run_in_executor(None, router.route, sym2),
        )

        results = []
        market = await loop.run_in_executor(None, get_market_filter)
        for sym, fr in [(sym1, r1), (sym2, r2)]:
            daily = add_indicators(fr.daily).dropna()
            h4    = add_indicators(fr.h4).dropna()
            h1    = add_indicators(fr.h1).dropna()
            setup = detect_swing_setup(daily, h4, h1, market)
            results.append((fr.symbol, setup))

        def badge(r: str) -> str:
            r = r.lower()
            if "high" in r: return "🟢"
            if "watch" in r: return "🟡"
            return "🔴"

        lines = [f"⚔️ {results[0][0]} vs {results[1][0]} 對比分析\n"]
        for sym, s in results:
            lines.append(
                f"{badge(s['rating'])} {sym}\n"
                f"   評級：{s['rating']}\n"
                f"   Setup：{s.get('setup_type','N/A')}\n"
                f"   進場：{s.get('planned_entry',0):.2f}\n"
                f"   止損：{s.get('stop_loss',0):.2f}\n"
                f"   目標：{s.get('target_1',0):.2f}\n"
                f"   風報比：{s.get('rr_ratio',0):.1f}R\n"
            )

        rr0, rr1 = results[0][1].get("rr_ratio", 0), results[1][1].get("rr_ratio", 0)
        r0_ok = not results[0][1].get("overheat_alert") and not results[0][1].get("storm_mode")
        r1_ok = not results[1][1].get("overheat_alert") and not results[1][1].get("storm_mode")

        if r0_ok and (not r1_ok or rr0 >= rr1):
            winner = results[0][0]
        elif r1_ok:
            winner = results[1][0]
        else:
            winner = None

        if winner:
            lines.append(f"👑 AI 優先推薦：{winner}（風報比較高 / 無熔斷）")
        else:
            lines.append("⚠️ 兩支目前均有熔斷或警示，建議等待更好時機")
        lines.append("\n輸入代號查看完整 Flex 分析（例：NVDA）")

        await push_line(user_id, "\n".join(lines))

    except Exception as e:
        print(f"❌ Compare 失敗: {e}")
        await push_line(user_id, "⚠️ 比較分析時發生錯誤，請稍後再試。")


async def run_chat_response(user_msg: str, user_id: str) -> None:
    """
    非代號的股票相關問題 → gpt-4o-mini 輕量回覆。
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.45,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 WengStock AI，專業的美股 Swing Trading 助理。\n"
                        "只回答股票和交易相關問題。短、直接、有交易員感。用繁體中文回答。\n\n"
                        "【重要行為規則】\n"
                        "1. 如果用戶問「推薦股票」、「幫我挑股」、「哪些可以買」、「掃描」，\n"
                        "   請回覆：「請輸入 /scan 美股 或 /scan 台股，我會掃描 watchlist 找出今日高勝率機會 🔍」\n"
                        "2. 如果用戶問「為什麼都是 No Trade」，\n"
                        "   請解釋：「No Trade 代表目前這檔股票不符合高勝率進場條件（可能超漲、大盤弱、風報比不足）。\n"
                        "   可輸入 /scan 美股 讓系統掃描整個 watchlist，找出目前有機會的標的。」\n"
                        "3. 不要給一般股市教育或理財建議，不要提經紀商帳戶問題。\n"
                        "4. 不保證獲利，注意風險。"
                    ),
                },
                {"role": "user", "content": user_msg},
            ],
        )
        reply = response.choices[0].message.content
        await push_line(user_id, f"🤖 WengStock AI：\n\n{reply}")

    except Exception as e:
        print(f"❌ Chat 回覆失敗: {e}")
        await push_line(user_id, "⚠️ 回覆時發生錯誤，請稍後再試。")


# ─────────────────────────────────────────────
# Prompt 建構
# ─────────────────────────────────────────────

def _build_analysis_prompt(symbol: str, setup: dict, news_text: str) -> str:
    storm_mode     = setup.get("storm_mode", False)
    overheat_alert = setup.get("overheat_alert", False)
    dist_alert     = setup.get("distribution_alert", False)
    exit_action    = setup.get("exit_action", "hold")
    exit_message   = setup.get("exit_message", "")
    dev_pct        = setup.get("deviation_from_ma20_pct", 0)
    trailing_stop  = setup.get("trailing_stop", setup["stop_loss"])
    stop_loss_price = setup.get("stop_loss_price", setup["stop_loss"])

    # 依觸發的防護邏輯，動態組裝強制指示段
    iron_rule_block = ""
    if storm_mode:
        iron_rule_block += (
            "\n⛔【風暴防禦模式啟動】⛔\n"
            "請第一段用大白話傳達：「現在市場正在刮颱風，外面很危險，"
            "AI 建議您現在抱著現金好好休息，等大晴天我們再出來開工！」\n"
            "不做進場分析，只做防禦建議。\n"
        )
    if overheat_alert and not storm_mode:
        iron_rule_block += (
            f"\n🔥【追高熔斷 — 正乖離 {dev_pct:.1f}%】\n"
            "請用大白話告訴用戶：「現在這檔股票太熱太貴了，"
            "我們去買會幫別人洗碗，請耐心等它降溫拉回再考慮！」\n"
        )
    if dist_alert:
        iron_rule_block += (
            f"\n🚨【大戶倒貨警報】請將以下訊息放在最顯眼位置：\n{exit_message}\n"
        )
    elif exit_action in ("partial_exit", "review"):
        iron_rule_block += f"\n📤【出場訊號 — {exit_action}】\n{exit_message}\n"

    return f"""
你是 WengStock AI，充滿良心、保護散戶與長輩資產的 Swing Trading 分析助理。
{iron_rule_block}
股票：{symbol}

══ 技術數據 ══
現價：{setup["price"]:.2f}  EMA20：{setup["ema20"]:.2f}  EMA50：{setup["ema50"]:.2f}
RSI：{setup["rsi"]:.2f}  ATR：{setup["atr"]:.2f}  量比：{setup["vol_ratio"]:.2f}
正乖離率（MA20）：{dev_pct:.1f}%
系統評級：{setup["rating"]}  Setup：{setup["setup_type"]}  大盤：{setup["market_status"]}

══ 進場計畫 ══
等待區：{setup["entry_zone_low"]:.2f} - {setup["entry_zone_high"]:.2f}
計畫進場：{setup["planned_entry"]:.2f}  偏離等待區：{setup["distance_from_entry"]*100:.1f}%

══ 鐵律三：止損絕對執行 ══
絕對止損價（2ATR below entry）：{stop_loss_price:.2f}
動態止損參考：{setup["stop_loss"]:.2f}
目標一：{setup["target_1"]:.2f}  風報比：{setup["rr_ratio"]:.2f}R

══ 出場分析 ══
動態防守價（Trailing Stop）：{trailing_stop:.2f}
出場動作：{exit_action}
出場訊息：{exit_message}

支持理由：{setup["reasons"]}
風險警告：{setup["warnings"]}

近期新聞：
{news_text}

══ 輸出格式（LINE 適合閱讀）══
📊 {symbol} Swing 分析

1. 方向 & 現況（大盤 + 個股趨勢）
2. Setup 品質（說明評級原因）
3. 進場策略（等待區、追價判斷）
4. 🛡️ 止損鐵律（必寫：「看錯不可恥，如果跌破 {stop_loss_price:.2f} 請一定要果斷賣掉，這是在保護我們的退休金！」）
5. 出場訊號（根據 exit_action 傳達對應的大白話出場建議）
6. 新聞面（簡短說是否影響 swing）
7. 最後一句紀律提醒

規則：不保證獲利、不說穩賺、如果是 🔴 評級不強調目標價。
"""


# ─────────────────────────────────────────────
# FastAPI Webhook 路由
# ─────────────────────────────────────────────

@app.get("/")
async def health() -> dict:
    return {"status": "WengStock AI v2 running", "time": datetime.now().isoformat()}


@app.get("/scheduler/status", summary="排程器狀態與時區診斷")
async def scheduler_status(request: Request) -> dict:
    """
    確認所有定時任務下次觸發時間，並顯示當前 DST 狀態。
    部署後可用這個端點驗證夏令/冬令時間是否正確。
    """
    sch = request.app.state.scheduler
    tz_info = get_market_tz_info()

    jobs = [
        {
            "id":       job.id,
            "name":     job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
        }
        for job in sch.get_jobs()
    ]

    return {
        "timezone": tz_info,
        "jobs":     jobs,
    }


@app.post("/callback")
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    for event in body.get("events", []):
        if event.get("type") != "message":
            continue
        if event.get("message", {}).get("type") != "text":
            continue

        user_msg = event["message"]["text"].strip()
        reply_token = event.get("replyToken", "")
        user_id = event.get("source", {}).get("userId", "")

        print(f"💬 [{user_id[:8]}...] {user_msg}")

        # ── 新用戶歡迎
        if is_new_user(user_id):
            mark_user_seen(user_id)
            await push_line(user_id, _welcome_text())

        # ── 快速命令（同步直接回覆）
        if user_msg.lower() in ["/help", "help", "幫助"]:
            await reply_line(reply_token, _help_text())
            continue

        if user_msg.lower() in ["/stats", "stats", "統計"]:
            await reply_line(reply_token, analyze_trade_history())
            continue

        if user_msg.lower() in ["/morning", "morning"]:
            await reply_line(reply_token, "📈 盤前簡報生成中，請稍候約 15 秒...")
            background_tasks.add_task(run_morning_brief, user_id)
            continue

        # ── /exit SYMBOL → 出場分析（背景）
        lower = user_msg.lower().strip()
        if lower.startswith("/exit ") or lower.startswith("exit "):
            parts = user_msg.split()
            if len(parts) >= 2 and is_direct_ticker(parts[1]):
                exit_sym = parts[1].upper()
                await reply_line(
                    reply_token,
                    f"📤 {exit_sym} 出場分析中...\n正在計算 Trailing Stop、倒貨訊號，請稍候約 10 秒。",
                )
                background_tasks.add_task(run_exit_analysis, exit_sym, user_id)
                continue

        # ── /size SYMBOL [AMOUNT] → 部位計算
        if lower.startswith("/size ") or lower.startswith("size "):
            parts = user_msg.split()
            if len(parts) >= 2 and is_direct_ticker(parts[1]):
                sym  = parts[1].upper()
                try:
                    amt = float(parts[2]) if len(parts) >= 3 else float(
                        os.getenv("DEFAULT_ACCOUNT_SIZE", "0")
                    )
                except ValueError:
                    amt = 0
                if amt <= 0:
                    await reply_line(reply_token,
                        "請告訴我您的帳戶資金 💰\n\n"
                        f"例如：/size {sym} 50000\n"
                        "（代表帳戶 5 萬元，系統自動算出最多買幾股）")
                    continue
                await reply_line(reply_token,
                    f"💰 {sym} 部位計算中，請稍候...")
                background_tasks.add_task(run_size_analysis, sym, amt, user_id)
                continue

        # ── /alert SYMBOL PRICE → 條件式價格警報
        if lower.startswith("/alert ") or lower.startswith("alert "):
            parts = user_msg.split()
            if len(parts) >= 3 and is_direct_ticker(parts[1]):
                sym = parts[1].upper()
                try:
                    target = float(parts[2])
                    await reply_line(reply_token, f"🔔 {sym} 警報設定中...")
                    background_tasks.add_task(run_alert_setup, sym, target, user_id)
                except ValueError:
                    await reply_line(reply_token,
                        "請輸入正確的目標價格 📊\n\n"
                        f"例如：/alert {sym} 850")
                continue

        # ── /myalerts → 查看我的警報
        if lower in ["/myalerts", "myalerts", "我的警報"]:
            await reply_line(reply_token, format_user_alerts(user_id))
            continue

        # ── /cancelalert SYMBOL → 取消警報
        if lower.startswith("/cancelalert ") or lower.startswith("cancelalert "):
            parts = user_msg.split()
            if len(parts) >= 2:
                sym = parts[1].upper()
                await reply_line(reply_token, remove_alert(user_id, sym))
            continue

        # ── /watch SYMBOL → 加入自選股
        if lower.startswith("/watch ") or lower.startswith("watch "):
            parts = user_msg.split()
            if len(parts) >= 2 and is_direct_ticker(parts[1]):
                sym = parts[1].upper()
                msg = add_watch(user_id, sym)
                earn = get_earnings_warning(sym)
                if earn:
                    msg += f"\n\n{earn}"
                await reply_line(reply_token, msg)
            else:
                await reply_line(reply_token,
                    "請輸入正確的股票代號 📋\n\n例如：/watch NVDA")
            continue

        # ── /unwatch SYMBOL → 移除自選股
        if lower.startswith("/unwatch ") or lower.startswith("unwatch "):
            parts = user_msg.split()
            if len(parts) >= 2:
                sym = parts[1].upper()
                await reply_line(reply_token, remove_watch(user_id, sym))
            continue

        # ── /mywatchlist → 查看自選股
        if lower in ["/mywatchlist", "mywatchlist", "我的自選股", "自選股"]:
            await reply_line(reply_token, format_user_watchlist(user_id))
            continue

        # ── /compare SYM1 SYM2 → 同板塊比較
        if lower.startswith("/compare ") or lower.startswith("compare "):
            parts = user_msg.split()
            if len(parts) >= 3 and is_direct_ticker(parts[1]) and is_direct_ticker(parts[2]):
                s1, s2 = parts[1].upper(), parts[2].upper()
                await reply_line(reply_token,
                    f"⚔️ 正在比較 {s1} vs {s2}，請稍候約 20 秒...")
                background_tasks.add_task(run_compare_analysis, s1, s2, user_id)
            else:
                await reply_line(reply_token,
                    "請輸入兩個股票代號 📊\n\n例如：/compare NVDA AMD")
            continue

        # ── /gsubscribe → 訂閱大猩猩每日掃描推播
        if lower.startswith("/gsubscribe") or lower == "gsubscribe":
            mkt = "BOTH"
            if "美股" in lower or " us" in lower:
                mkt = "US"
            elif "台股" in lower or " tw" in lower:
                mkt = "TW"
            await reply_line(reply_token, gorilla_subscribe(user_id, mkt))
            continue

        # ── /gunsubscribe → 取消訂閱
        if lower.startswith("/gunsubscribe") or lower == "gunsubscribe":
            await reply_line(reply_token, gorilla_unsubscribe(user_id))
            continue

        # ── /gorilla SYMBOL → 大猩猩診斷（背景）
        if lower.startswith("/gorilla ") or lower.startswith("gorilla "):
            parts = user_msg.split()
            if len(parts) >= 2:
                g_sym = parts[1].upper()
                flag  = "🇹🇼" if g_sym.isdigit() else "🇺🇸"
                await reply_line(reply_token,
                    f"🦍 大猩猩診斷 {flag} {g_sym} 中...\n"
                    "正在抓基本面 + 技術面數據，請稍候約 20 秒。")
                background_tasks.add_task(run_gorilla_diagnosis, g_sym, user_id)
            else:
                await reply_line(reply_token,
                    "請輸入股票代號 🦍\n\n"
                    "美股：/gorilla NVDA\n"
                    "台股：/gorilla 2330")
            continue

        # ── /scan [gorilla] [美股|台股|US|TW] → 掃描 watchlist
        if lower.startswith("/scan") or lower.startswith("scan ") or lower in ["scan", "掃描", "掃一下"]:
            parts = lower.split()
            if len(parts) > 1 and parts[1] == "gorilla":
                mkt   = "TW" if (len(parts) > 2 and parts[2] in ["台股", "tw", "台灣"]) else "US"
                label = "台股" if mkt == "TW" else "美股"
                await reply_line(reply_token,
                    f"🦍 大猩猩策略掃描 {label} 中...\n"
                    "CAN SLIM 篩選約 30~60 秒，請稍候。")
                background_tasks.add_task(run_gorilla_scan, mkt, user_id)
            else:
                mkt       = "TW" if (len(parts) > 1 and parts[1] in ["台股", "tw", "台灣"]) else "US"
                mkt_label = "台股" if mkt == "TW" else "美股"
                await reply_line(reply_token,
                    f"🔍 正在掃描 {mkt_label} watchlist，找出今日高勝率機會，請稍候約 30 秒...")
                background_tasks.add_task(run_scan_analysis, mkt, user_id)
            continue

        # ── /gentry SYMBOL PRICE → 記錄大猩猩進場
        if lower.startswith("/gentry ") or lower.startswith("gentry "):
            parts = user_msg.split()
            if len(parts) >= 3:
                g_sym = parts[1].upper()
                try:
                    entry_price = float(parts[2])
                    is_tw       = g_sym.isdigit()
                    msg         = record_entry(user_id, g_sym, entry_price, is_tw)
                    await reply_line(reply_token, msg)
                except ValueError:
                    await reply_line(reply_token,
                        "請輸入正確的進場價格 💰\n\n"
                        "例如：/gentry NVDA 850.00\n"
                        "      /gentry 2330 980")
            else:
                await reply_line(reply_token,
                    "格式：/gentry 股票代號 進場價格\n\n"
                    "例如：/gentry NVDA 850\n"
                    "      /gentry 2330 980")
            continue

        # ── /gexit SYMBOL → 大猩猩標記出場
        if lower.startswith("/gexit ") or lower.startswith("gexit "):
            parts = user_msg.split()
            if len(parts) >= 2:
                g_sym = parts[1].upper()
                await reply_line(reply_token, close_position(user_id, g_sym))
            else:
                await reply_line(reply_token,
                    "格式：/gexit 股票代號\n\n例如：/gexit NVDA")
            continue

        # ── /gpositions → 查看大猩猩持倉（yfinance 同步，丟 executor）
        if lower in ["/gpositions", "gpositions", "持倉", "大猩猩持倉"]:
            loop = asyncio.get_event_loop()
            pos_text = await loop.run_in_executor(None, format_positions, user_id)
            await reply_line(reply_token, pos_text)
            continue

        # ── /gmarket → 大盤風向球（yfinance 同步，丟 executor）
        if lower in ["/gmarket", "gmarket", "大盤", "市場狀態"]:
            loop = asyncio.get_event_loop()
            mkt_text = await loop.run_in_executor(None, format_market_status)
            await reply_line(reply_token, mkt_text)
            continue

        # ── 非股票問題 → 拒絕
        if not is_stock_related(user_msg):
            await reply_line(
                reply_token,
                "我是 Swing Trading 專家，只能回答股票相關問題 📈\n"
                "輸入 /help 查看功能。",
            )
            continue

        # ── 股票代號 → 背景深度分析（gpt-4o）
        if is_direct_ticker(user_msg):
            await reply_line(
                reply_token,
                f"⏳ WengStock 收到！\n正在為您掃描 {user_msg.upper()} 的波段型態與最新消息，請稍候約 15 秒...",
            )
            background_tasks.add_task(run_swing_analysis, user_msg, user_id)
            continue

        # ── 股票相關問題 → 背景輕量回覆（gpt-4o-mini）
        await reply_line(reply_token, "⏳ 思考中，請稍候...")
        background_tasks.add_task(run_chat_response, user_msg, user_id)

    return JSONResponse(content={"status": "ok"})


# ─────────────────────────────────────────────
# Help 文字
# ─────────────────────────────────────────────

def _welcome_text() -> str:
    return """
👋 歡迎使用 WengStock AI！

我是你的 Swing Trading 專屬助理，幫你找進場機會、控制風險、守住資產。

快速開始：
  📊 輸入股票代號 → NVDA
  🔍 掃描今日機會 → /scan 美股
  📋 加入自選股   → /watch NVDA

輸入 /help 查看所有功能。

⚠️ 我不是喊單工具，每個決定請自行判斷風險。
""".strip()


def _help_text() -> str:
    return """
🤖 WengStock AI v2

📊 完整分析：直接輸入股票代號
  NVDA / TSLA / AMD / 2330

🔍 掃描推薦（幫你找今日機會）：
  /scan 美股
  /scan 台股

⚔️ 同板塊比較：
  /compare NVDA AMD

📋 個人自選股（每日自動推播）：
  /watch NVDA      加入自選股
  /unwatch NVDA    移除自選股
  /mywatchlist     查看清單

📤 出場分析（手上有股票要不要賣）：
  /exit NVDA

💰 部位計算（我該買幾股？）：
  /size NVDA 50000

🔔 條件式警報（到價通知）：
  /alert NVDA 850
  /myalerts        查看我的警報
  /cancelalert NVDA 取消警報

🦍 大猩猩策略（CAN SLIM 成長選股）：
  /gorilla NVDA       診斷單支股票
  /gorilla 2330       台股診斷
  /scan gorilla       掃描美股精選
  /scan gorilla 台股  掃描台股精選
  /gmarket            大盤風向球

🦍 大猩猩持倉管理：
  /gentry NVDA 850  記錄進場（自動設停損）
  /gexit NVDA       標記出場
  /gpositions       查看所有持倉狀態

🔔 大猩猩每日推播訂閱：
  /gsubscribe       訂閱每日掃描（美股＋台股）
  /gsubscribe 美股  只訂美股
  /gsubscribe 台股  只訂台股
  /gunsubscribe     取消訂閱

💬 聊天提問：
  TSLA 現在能追嗎？
  大盤今天怎麼看？

📈 /morning  盤前簡報
📊 /stats    交易統計
❓ /help     這份說明

🛡️ 三條鐵律永遠生效：
  • 正乖離 > 15% 強制熔斷
  • 大盤跌破 MA200 啟動防禦
  • 每次進場都給你精準止損價

⚠️ 我不是喊單工具，我幫你守住退休金。
""".strip()
