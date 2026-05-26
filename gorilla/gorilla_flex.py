"""
大猩猩策略 — LINE Flex Message 卡片
Bloomberg 暗色風格，與現有 WengStock Swing 卡片視覺統一。
"""
from __future__ import annotations

_SIGNAL_PALETTE = {
    "BUY":         {"badge": "🦍 BUY 試單",   "color": "#00E676", "bg": "#0A2E1A"},
    "ADD":         {"badge": "📈 加碼訊號",    "color": "#FFD600", "bg": "#2E2800"},
    "STOP_LOSS":   {"badge": "🛑 停損出場",    "color": "#FF5252", "bg": "#2E0A0A"},
    "TAKE_PROFIT": {"badge": "🎯 移動停利",    "color": "#00BFA5", "bg": "#002E2A"},
    "DIAGNOSIS":   {"badge": "🔬 大猩猩診斷",  "color": "#82B1FF", "bg": "#0A1A2E"},
    "NO_PASS":     {"badge": "❌ 不符條件",    "color": "#FF5252", "bg": "#2E0A0A"},
}


def _kv(key: str, val: str, color: str = "#E0E0E0") -> dict:
    return {
        "type": "box", "layout": "horizontal",
        "contents": [
            {"type": "text", "text": key,  "color": "#808080", "size": "sm", "flex": 5},
            {"type": "text", "text": val,  "color": color,     "size": "sm",
             "align": "end", "flex": 5, "weight": "bold", "wrap": True},
        ],
        "margin": "sm",
    }


def _divider() -> dict:
    return {"type": "separator", "color": "#2A2A3A", "margin": "md"}


def _label(text: str) -> dict:
    return {"type": "text", "text": text, "color": "#606080", "size": "xs", "margin": "md"}


def build_gorilla_flex(signal_type: str, result: dict) -> dict:
    """
    signal_type: BUY / ADD / STOP_LOSS / TAKE_PROFIT / DIAGNOSIS / NO_PASS
    result: dict from screener.screen_us / screen_tw or position_manager
    """
    palette = _SIGNAL_PALETTE.get(signal_type, _SIGNAL_PALETTE["DIAGNOSIS"])
    ticker  = result.get("ticker", "N/A")
    price   = result.get("price", 0)
    flag    = "🇹🇼" if result.get("is_tw") else "🇺🇸"

    # ── Header ──────────────────────────────────
    header = {
        "type": "box", "layout": "vertical",
        "backgroundColor": "#0D1117", "paddingAll": "lg",
        "contents": [
            {
                "type": "box", "layout": "horizontal",
                "contents": [
                    {"type": "text", "text": f"{flag} {ticker}",
                     "color": "#FFFFFF", "size": "xxl", "weight": "bold", "flex": 7},
                    {"type": "text", "text": "GORILLA",
                     "color": "#606080", "size": "xs", "align": "end",
                     "flex": 3, "gravity": "bottom"},
                ],
            },
            {
                "type": "box", "layout": "vertical",
                "backgroundColor": palette["bg"], "cornerRadius": "md",
                "paddingAll": "sm", "margin": "md",
                "contents": [{
                    "type": "text", "text": palette["badge"],
                    "color": palette["color"], "size": "md",
                    "weight": "bold", "align": "center",
                }],
            },
        ],
    }

    # ── Body ────────────────────────────────────
    body_contents: list[dict] = []

    # 基本面區塊
    rev = result.get("revenue_yoy")
    eps = result.get("eps_yoy")
    l2p = result.get("loss_to_profit", False)
    gm  = result.get("gross_margin_current")
    gmc = result.get("gross_margin_change")
    peg = result.get("peg")
    csq = result.get("consecutive_buy")   # 台股外資連買

    if any(v is not None for v in [rev, eps, gm]):
        body_contents.append(_label("▸ 基本面"))
        if rev is not None:
            body_contents.append(_kv(
                "營收 YoY" if not result.get("is_tw") else "月營收 YoY",
                f"{rev:.1f}%",
                "#00E676" if rev >= (25 if not result.get("is_tw") else 20) else "#FF5252",
            ))
        if l2p:
            body_contents.append(_kv("EPS", "虧轉盈 ✅", "#00E676"))
        elif eps is not None:
            body_contents.append(_kv(
                "EPS YoY",
                f"{eps:.1f}%",
                "#00E676" if eps >= 20 else "#FF5252",
            ))
        if gm is not None:
            body_contents.append(_kv(
                "毛利率",
                f"{gm:.1f}%（{gmc:+.1f}%）" if gmc is not None else f"{gm:.1f}%",
                "#00E676" if (gmc or 0) >= 0 else "#FF5252",
            ))
        if peg is not None:
            body_contents.append(_kv("PEG", f"{peg:.2f}",
                                      "#00E676" if peg < 1.2 else "#FF5252"))
        if csq is not None:
            body_contents.append(_kv("外資連買", f"{csq} 天",
                                      "#00E676" if csq >= 3 else "#FF9800"))
        body_contents.append(_divider())

    # 技術面區塊
    body_contents.append(_label("▸ 技術面"))
    body_contents.append(_kv("現價", f"${price:,.2f}", "#FFFFFF"))

    sma50  = result.get("sma50")
    sma200 = result.get("sma200")
    ma60   = result.get("ma60")
    ma240  = result.get("ma240")
    vr     = result.get("vol_ratio", 0)

    if sma50:
        body_contents.append(_kv("50 SMA", f"{sma50:,.2f}",
                                  "#00E676" if price > sma50 else "#FF5252"))
    if sma200:
        body_contents.append(_kv("200 SMA", f"{sma200:,.2f}",
                                  "#00E676" if (sma50 or 0) > sma200 else "#FF5252"))
    if ma60:
        body_contents.append(_kv("60 MA", f"{ma60:,.2f}",
                                  "#00E676" if price > ma60 else "#FF5252"))
    if ma240:
        body_contents.append(_kv("240 MA", f"{ma240:,.2f}",
                                  "#00E676" if (ma60 or 0) > ma240 else "#FF5252"))
    body_contents.append(_kv("成交量比", f"{vr:.1f}x",
                              "#00E676" if vr >= 1.5 else "#FF9800"))
    body_contents.append(_divider())

    # 操作建議區塊（BUY / ADD）
    if signal_type == "BUY":
        stop = round(price * (1 - 0.075), 2)
        tgt1 = round(price * 1.20, 2)
        tgt2 = round(price * 1.40, 2)
        body_contents += [
            _label("▸ 操作建議"),
            _kv("建議配置", "5% 試單", "#FFD600"),
            _kv("🛑 鐵血停損", f"${stop:,.2f}（-7.5%）", "#FF5252"),
            _kv("目標一",     f"${tgt1:,.2f}（+20%）",   "#00E676"),
            _kv("目標二",     f"${tgt2:,.2f}（+40%）",   "#00BFA5"),
            _divider(),
        ]
    elif signal_type == "ADD":
        body_contents += [
            _label("▸ 加碼建議"),
            _kv("加碼配置", "3%（正金字塔）", "#FFD600"),
            _kv("加碼價位", f"${price:,.2f}",  "#FFFFFF"),
            _divider(),
        ]

    # 評分明細
    passes = result.get("passes", [])
    fails  = result.get("fails", [])
    if passes or fails:
        body_contents.append(_label("▸ 評分明細"))
        for p in passes[:5]:
            body_contents.append({
                "type": "text", "text": f"  ✅ {p}",
                "color": "#00E676", "size": "xs", "margin": "xs", "wrap": True,
            })
        for f in fails[:5]:
            body_contents.append({
                "type": "text", "text": f"  ❌ {f}",
                "color": "#FF5252", "size": "xs", "margin": "xs", "wrap": True,
            })

    body = {
        "type": "box", "layout": "vertical",
        "backgroundColor": "#0D1117", "paddingAll": "lg",
        "contents": body_contents,
    }

    footer = {
        "type": "box", "layout": "vertical",
        "backgroundColor": "#111827", "paddingAll": "md",
        "contents": [{
            "type": "text",
            "text": "WengStock AI 大猩猩策略 · 非投資建議 · 嚴守停損",
            "color": "#404060", "size": "xxs", "align": "center",
        }],
    }

    bubble = {
        "type": "bubble", "size": "mega",
        "header": header,
        "body":   body,
        "footer": footer,
        "styles": {
            "header": {"backgroundColor": "#0D1117"},
            "body":   {"backgroundColor": "#0D1117"},
            "footer": {"backgroundColor": "#111827"},
        },
    }

    return {
        "type":     "flex",
        "altText":  f"🦍 {ticker} 大猩猩訊號 — {palette['badge']}",
        "contents": bubble,
    }
