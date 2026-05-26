"""
大猩猩策略 — 持倉風控管理
鐵血停損 7.5%、正金字塔加碼、移動停利（20% 啟動）
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import yfinance as yf

_POS_FILE      = Path(__file__).parent.parent / "gorilla_positions.json"
HARD_STOP_PCT  = 0.075   # -7.5% 鐵血停損
TRAIL_TRIGGER  = 0.20    # 帳面 +20% → 啟動移動停利
TRAIL_DRAWDOWN = 0.10    # 從波段高點回檔 10% → 出場
ADD_MIN_PROFIT = 0.05    # 至少獲利 5% 才考慮加碼


# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────

def _load() -> dict:
    try:
        return json.loads(_POS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    _POS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────────────────────
# 進場記錄
# ─────────────────────────────────────────────

def record_entry(user_id: str, ticker: str, entry_price: float, is_tw: bool = False) -> str:
    data = _load()
    user = data.setdefault(user_id, {})

    if ticker in user and user[ticker].get("status") == "open":
        return f"⚠️ {ticker} 已有開倉記錄，請先 /gexit {ticker} 結清再重新進場"

    stop = round(entry_price * (1 - HARD_STOP_PCT), 2)
    user[ticker] = {
        "entry_price":     entry_price,
        "entry_date":      datetime.now().strftime("%Y-%m-%d"),
        "allocation_pct":  5.0,
        "stop_price":      stop,
        "peak_price":      entry_price,
        "trailing_active": False,
        "added":           False,
        "add_price":       None,
        "status":          "open",
        "is_tw":           is_tw,
    }
    _save(data)
    return (
        f"✅ {ticker} 進場記錄已建立\n"
        f"進場價：{entry_price:.2f}\n"
        f"🛑 鐵血停損線：{stop:.2f}（-7.5%）\n"
        f"💼 建議配置：帳戶 5%（試單）"
    )


def close_position(user_id: str, ticker: str, reason: str = "手動出場") -> str:
    data = _load()
    pos  = data.get(user_id, {}).get(ticker)
    if not pos or pos.get("status") != "open":
        return f"❌ 找不到 {ticker} 的開倉記錄"
    pos["status"]     = "closed"
    pos["close_date"] = datetime.now().strftime("%Y-%m-%d")
    pos["close_note"] = reason
    _save(data)
    return f"✅ {ticker} 已標記出場（{reason}）"


# ─────────────────────────────────────────────
# 查詢格式化
# ─────────────────────────────────────────────

def format_positions(user_id: str) -> str:
    data   = _load()
    open_p = {t: p for t, p in data.get(user_id, {}).items() if p.get("status") == "open"}

    if not open_p:
        return (
            "📭 目前無開倉記錄\n\n"
            "記錄進場：/gentry NVDA 850\n"
            "（系統自動設定停損線並每日監控）"
        )

    lines = ["📊 大猩猩持倉狀態：\n"]
    for ticker, pos in open_p.items():
        suffix = ".TW" if pos.get("is_tw") else ""
        try:
            hist  = yf.Ticker(f"{ticker}{suffix}").history(period="2d", interval="1d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else 0
        except Exception:
            price = 0

        profit    = (price - pos["entry_price"]) / pos["entry_price"] * 100 if price else 0
        p_emoji   = "🟢" if profit >= 0 else "🔴"
        trail_tag = " 🎯移動停利中" if pos["trailing_active"] else ""
        add_tag   = " ✅已加碼" if pos["added"] else ""

        lines.append(
            f"{p_emoji} {ticker}{trail_tag}{add_tag}\n"
            f"   進場：{pos['entry_price']:.2f} → 現價：{price:.2f}（{profit:+.1f}%）\n"
            f"   停損：{pos['stop_price']:.2f}"
        )

    lines.append("\n查看大盤風向：/gmarket")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 每日自動風控檢查（Scheduler 呼叫）
# ─────────────────────────────────────────────

def check_positions_sync(user_id: str) -> list[dict]:
    """
    同步版本（在 executor 中執行）。
    回傳需要推播的警報 list[{type, ticker, msg}]。
    """
    data      = _load()
    positions = data.get(user_id, {})
    alerts: list[dict] = []
    changed   = False

    for ticker, pos in positions.items():
        if pos.get("status") != "open":
            continue

        suffix = ".TW" if pos.get("is_tw") else ""
        try:
            hist  = yf.Ticker(f"{ticker}{suffix}").history(period="30d", interval="1d")
            if hist.empty or len(hist) < 2:
                continue
            price  = float(hist["Close"].iloc[-1])
            prev   = float(hist["Close"].iloc[-2])
            ma20   = float(hist["Close"].rolling(20).mean().iloc[-1]) if len(hist) >= 20 else None
        except Exception:
            continue

        entry  = pos["entry_price"]
        profit = (price - entry) / entry

        # 更新波段高點
        if price > pos["peak_price"]:
            pos["peak_price"] = price
            changed = True

        # ① 鐵血停損
        if price <= pos["stop_price"]:
            pos["status"] = "stopped"
            changed = True
            alerts.append({
                "type":   "STOP_LOSS",
                "ticker": ticker,
                "price":  price,
                "msg": (
                    f"🛑 停損警報 — {ticker}\n"
                    f"現價 {price:.2f} 跌破停損線 {pos['stop_price']:.2f}\n"
                    f"虧損：{profit*100:.1f}%\n"
                    "請立刻出場，嚴守紀律！"
                ),
            })
            continue

        # ② 啟動移動停利
        if profit >= TRAIL_TRIGGER and not pos["trailing_active"]:
            pos["trailing_active"] = True
            changed = True

        # ③ 移動停利觸發
        if pos["trailing_active"]:
            drawdown    = (pos["peak_price"] - price) / pos["peak_price"]
            below_ma20  = ma20 is not None and price < ma20
            if drawdown >= TRAIL_DRAWDOWN or below_ma20:
                reason = "收盤跌破 20MA" if below_ma20 else f"從高點回檔 {drawdown*100:.1f}%"
                pos["status"] = "exited"
                changed = True
                alerts.append({
                    "type":   "TAKE_PROFIT",
                    "ticker": ticker,
                    "price":  price,
                    "msg": (
                        f"🎯 移動停利 — {ticker}\n"
                        f"觸發：{reason}\n"
                        f"帳面獲利：+{profit*100:.1f}%\n"
                        "建議全數出場，鎖定利潤！"
                    ),
                })
                continue

        # ④ 正金字塔加碼機會（回測 20MA 翻揚）
        if not pos["added"] and profit >= ADD_MIN_PROFIT and ma20:
            if prev <= ma20 and price > ma20:
                pos["added"]    = True
                pos["add_price"] = price
                changed = True
                alerts.append({
                    "type":   "ADD",
                    "ticker": ticker,
                    "price":  price,
                    "msg": (
                        f"📈 加碼訊號 — {ticker}\n"
                        f"回測 20MA（{ma20:.2f}）後翻揚！\n"
                        f"現價：{price:.2f}，帳面獲利：+{profit*100:.1f}%\n"
                        "建議追加 3% 倉位（正金字塔加碼）"
                    ),
                })

    if changed:
        data[user_id] = positions
        _save(data)

    return alerts


def get_all_user_ids() -> list[str]:
    return [uid for uid in _load().keys()]
