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
    is_stock_related,
    is_direct_ticker,
    is_industry_question,
    log_trade,
    analyze_trade_history,
    calc_position_size,
    format_position_size,
)
from alerts import add_alert, remove_alert, format_user_alerts
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

    except asyncio.TimeoutError:
        await push_line(user_id, "⚠️ 分析超時，請稍後再試。")
    except Exception as e:
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
                        "你是 WengStock AI，專業的美股 Swing Trading 助理。"
                        "只回答股票和交易相關問題。短、直接、有交易員感。"
                        "用繁體中文回答。不保證獲利。"
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

        # ── 快速命令（同步直接回覆）
        if user_msg.lower() in ["/help", "help", "幫助"]:
            await reply_line(reply_token, _help_text())
            continue

        if user_msg.lower() in ["/stats", "stats", "統計"]:
            await reply_line(reply_token, analyze_trade_history())
            continue

        if user_msg.lower() in ["/morning", "morning"]:
            await reply_line(reply_token, "📈 盤前簡報生成中，請稍候...")
            # morning briefing 也可以丟背景，這裡略
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

def _help_text() -> str:
    return """
🤖 WengStock AI v2

📊 完整分析：直接輸入股票代號
  NVDA / TSLA / AMD / 2330

📤 出場分析（手上有股票要不要賣）：
  /exit NVDA

💰 部位計算（我該買幾股？）：
  /size NVDA 50000

🔔 條件式警報（到價通知）：
  /alert NVDA 850
  /myalerts        查看我的警報
  /cancelalert NVDA 取消警報

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
