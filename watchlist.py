"""
WengStock AI — 個人自選股系統
純資料層：watchlist CRUD + 新用戶追蹤
"""
from __future__ import annotations

import json
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_WL_FILE   = os.path.join(_DIR, "watchlist.json")
_SEEN_FILE = os.path.join(_DIR, "seen_users.json")


# ─────────────────────────────────────────────
# 內部 I/O
# ─────────────────────────────────────────────

def _load_wl() -> dict:
    try:
        with open(_WL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_wl(data: dict) -> None:
    with open(_WL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_seen() -> list:
    try:
        with open(_SEEN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_seen(seen: list) -> None:
    with open(_SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(seen, f)


# ─────────────────────────────────────────────
# 自選股 CRUD
# ─────────────────────────────────────────────

def add_watch(user_id: str, symbol: str) -> str:
    data      = _load_wl()
    user_list = data.get(user_id, [])
    symbol    = symbol.upper()
    if symbol in user_list:
        return f"📌 {symbol} 已在您的自選股清單中"
    if len(user_list) >= 10:
        return "❌ 自選股最多 10 支，請先 /unwatch 移除一支"
    user_list.append(symbol)
    data[user_id] = user_list
    _save_wl(data)
    return (
        f"✅ 已加入自選股：{symbol}\n"
        "📨 每日盤前/收盤將自動推播分析報告"
    )


def remove_watch(user_id: str, symbol: str) -> str:
    data      = _load_wl()
    user_list = data.get(user_id, [])
    symbol    = symbol.upper()
    if symbol not in user_list:
        return f"❌ {symbol} 不在您的自選股清單中"
    user_list.remove(symbol)
    data[user_id] = user_list
    _save_wl(data)
    return f"🗑️ {symbol} 已從自選股移除"


def get_user_watchlist(user_id: str) -> list[str]:
    return _load_wl().get(user_id, [])


def format_user_watchlist(user_id: str) -> str:
    symbols = get_user_watchlist(user_id)
    if not symbols:
        return (
            "📭 您目前沒有自選股\n\n"
            "加入方式：/watch NVDA\n"
            "（加入後每日盤前/收盤自動收到分析報告）"
        )
    lines = ["📋 您的自選股：\n"]
    for s in symbols:
        lines.append(f"  • {s}")
    lines.append(f"\n共 {len(symbols)}/10 支")
    lines.append("移除：/unwatch NVDA")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Scheduler 用的查詢介面（替換 db_get_watchlist）
# ─────────────────────────────────────────────

def get_watchlist_by_market(market_type: str) -> list[dict]:
    """
    回傳 [{line_user_id, stock_code}]，供 scheduler 任務使用。
    market_type: "TW" → 4 位數代號；"US" → 英文代號
    """
    data  = _load_wl()
    pairs = []
    for uid, symbols in data.items():
        for sym in symbols:
            is_tw = sym.isdigit()
            if (market_type == "TW" and is_tw) or (market_type == "US" and not is_tw):
                pairs.append({"line_user_id": uid, "stock_code": sym})
    return pairs


def get_all_symbols_by_market(market_type: str) -> list[str]:
    """回傳特定市場所有追蹤代號（去重）"""
    return list({r["stock_code"] for r in get_watchlist_by_market(market_type)})


def get_users_tracking(stock_code: str) -> list[str]:
    """回傳追蹤特定代號的所有 user_id"""
    data = _load_wl()
    result = []
    for uid, symbols in data.items():
        if stock_code.upper() in symbols:
            result.append(uid)
    return result


# ─────────────────────────────────────────────
# 新用戶追蹤
# ─────────────────────────────────────────────

def is_new_user(user_id: str) -> bool:
    return user_id not in _load_seen()


def mark_user_seen(user_id: str) -> None:
    seen = _load_seen()
    if user_id not in seen:
        seen.append(user_id)
        _save_seen(seen)
