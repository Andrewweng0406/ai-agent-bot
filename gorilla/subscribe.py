"""大猩猩策略 — 用戶訂閱管理（每日掃描推播）"""
from __future__ import annotations

import json
from pathlib import Path

_SUB_FILE = Path(__file__).parent.parent / "gorilla_subscribers.json"


def _load() -> dict:
    try:
        return json.loads(_SUB_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"US": [], "TW": []}


def _save(data: dict) -> None:
    _SUB_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def subscribe(user_id: str, market: str = "BOTH") -> str:
    data = _load()
    targets = ["US", "TW"] if market == "BOTH" else [market]
    added: list[str] = []
    for m in targets:
        lst = data.setdefault(m, [])
        if user_id not in lst:
            lst.append(user_id)
            added.append("美股" if m == "US" else "台股")
    _save(data)
    if added:
        return (
            f"✅ 已訂閱大猩猩每日掃描（{'、'.join(added)}）\n\n"
            "每日收盤後系統自動 CAN SLIM 掃描，\n"
            "找到精選標的就直接推播給你！\n\n"
            "取消訂閱：/gunsubscribe"
        )
    return "✅ 您已是訂閱用戶，每日掃描會自動推播。"


def unsubscribe(user_id: str) -> str:
    data = _load()
    removed: list[str] = []
    for m in ["US", "TW"]:
        lst = data.get(m, [])
        if user_id in lst:
            lst.remove(user_id)
            removed.append("美股" if m == "US" else "台股")
    _save(data)
    if removed:
        return f"✅ 已取消大猩猩訂閱（{'、'.join(removed)}）"
    return "ℹ️ 您原本未訂閱大猩猩掃描。"


def get_subscribers(market: str) -> list[str]:
    """回傳訂閱指定市場的用戶 ID 列表（去重）。"""
    data = _load()
    return list(set(data.get(market, [])))


def is_subscribed(user_id: str, market: str = "US") -> bool:
    return user_id in get_subscribers(market)
