"""
WengStock AI — 條件式價格警報系統
純資料層：負責警報的 CRUD 與觸發判斷，不做任何網路請求。
"""
from __future__ import annotations

import json
import os
from datetime import datetime

ALERTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts.json")


# ─────────────────────────────────────────────
# 內部 I/O
# ─────────────────────────────────────────────

def _load() -> list[dict]:
    try:
        with open(ALERTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(alerts: list[dict]) -> None:
    with open(ALERTS_FILE, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# 公開 API
# ─────────────────────────────────────────────

def add_alert(user_id: str, symbol: str, target_price: float, current_price: float) -> str:
    """
    新增或更新警報。
    自動判斷方向：
      current > target → 等待下跌（direction="below"）
      current < target → 等待上漲（direction="above"）
    """
    alerts = _load()
    direction      = "below" if current_price > target_price else "above"
    direction_text = "跌到" if direction == "below" else "漲到"

    for a in alerts:
        if a["user_id"] == user_id and a["symbol"] == symbol and not a.get("triggered"):
            a.update({
                "target_price":  target_price,
                "direction":     direction,
                "set_at_price":  current_price,
                "created_at":    datetime.now().isoformat(),
            })
            _save(alerts)
            return f"✅ {symbol} 警報已更新：{direction_text} {target_price:.2f} 時通知您"

    alerts.append({
        "user_id":       user_id,
        "symbol":        symbol,
        "target_price":  target_price,
        "direction":     direction,
        "set_at_price":  current_price,
        "created_at":    datetime.now().isoformat(),
        "triggered":     False,
    })
    _save(alerts)
    return (
        f"🔔 已設定警報：{symbol} {direction_text} {target_price:.2f} 元時通知您！\n"
        f"（目前價格：{current_price:.2f}）"
    )


def remove_alert(user_id: str, symbol: str) -> str:
    alerts = _load()
    before  = len(alerts)
    alerts  = [a for a in alerts if not (a["user_id"] == user_id and a["symbol"] == symbol)]
    if len(alerts) < before:
        _save(alerts)
        return f"🗑️ {symbol} 警報已取消"
    return f"❌ 找不到 {symbol} 的警報"


def get_user_alerts(user_id: str) -> list[dict]:
    return [a for a in _load() if a["user_id"] == user_id and not a.get("triggered")]


def format_user_alerts(user_id: str) -> str:
    alerts = get_user_alerts(user_id)
    if not alerts:
        return (
            "📭 您目前沒有設定任何價格警報。\n\n"
            "設定方式：/alert NVDA 850\n"
            "（到達目標價時自動通知您）"
        )
    lines = ["🔔 您的價格警報：\n"]
    for a in alerts:
        direction_text = "跌到" if a["direction"] == "below" else "漲到"
        lines.append(f"  {a['symbol']} — {direction_text} {a['target_price']:.2f}")
    lines.append("\n取消警報：/cancelalert NVDA")
    return "\n".join(lines)


def scan_and_mark_triggered(current_prices: dict[str, float]) -> list[dict]:
    """
    傳入 {symbol: current_price}，回傳被觸發的警報列表，並在 alerts.json 標記已觸發。
    觸發條件：
      direction=below → current_price <= target_price
      direction=above → current_price >= target_price
    """
    alerts    = _load()
    triggered = []

    for alert in alerts:
        if alert.get("triggered"):
            continue
        sym = alert["symbol"]
        if sym not in current_prices:
            continue
        current   = current_prices[sym]
        target    = alert["target_price"]
        direction = alert.get("direction", "below")

        hit = (direction == "below" and current <= target) or \
              (direction == "above" and current >= target)

        if hit:
            alert["triggered"]       = True
            alert["triggered_at"]    = datetime.now().isoformat()
            alert["triggered_price"] = current
            triggered.append(alert)

    _save(alerts)
    return triggered


def get_all_active_symbols() -> list[str]:
    """回傳所有活躍警報的股票代號（去重），供 scheduler 批次抓價格。"""
    return list({a["symbol"] for a in _load() if not a.get("triggered")})
