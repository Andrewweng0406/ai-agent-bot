"""
LINE Flex Message builder — Bloomberg dark-mode terminal style.
"""

from __future__ import annotations


# ─────────────────────────────────────────────
# 評級配色
# ─────────────────────────────────────────────

_RATING_PALETTE: dict[str, dict[str, str]] = {
    "high":  {"text": "#00E676", "bg": "#0A2E1A", "badge": "🟢 High Quality"},
    "watch": {"text": "#FFD600", "bg": "#2E2800", "badge": "🟡 Watchlist"},
    "no":    {"text": "#FF5252", "bg": "#2E0A0A", "badge": "🔴 No Trade"},
}


def _resolve_palette(rating: str) -> dict[str, str]:
    r = rating.lower()
    if "high" in r:
        return _RATING_PALETTE["high"]
    if "watch" in r:
        return _RATING_PALETTE["watch"]
    return _RATING_PALETTE["no"]


# ─────────────────────────────────────────────
# 小工具
# ─────────────────────────────────────────────

def _kv_row(key: str, value: str, value_color: str = "#E0E0E0") -> dict:
    """一行 Key-Value，左右對齊。"""
    return {
        "type": "box",
        "layout": "horizontal",
        "contents": [
            {
                "type": "text",
                "text": key,
                "color": "#808080",
                "size": "sm",
                "flex": 4,
            },
            {
                "type": "text",
                "text": value,
                "color": value_color,
                "size": "sm",
                "align": "end",
                "flex": 6,
                "weight": "bold",
            },
        ],
        "margin": "sm",
    }


def _divider() -> dict:
    return {"type": "separator", "color": "#2A2A3A", "margin": "md"}


def _section_label(text: str) -> dict:
    return {
        "type": "text",
        "text": text,
        "color": "#606080",
        "size": "xs",
        "margin": "md",
    }


# ─────────────────────────────────────────────
# 主函數
# ─────────────────────────────────────────────

# 出場動作配色
_EXIT_PALETTE: dict[str, dict[str, str]] = {
    "hold":         {"color": "#00E676", "bg": "#0A2E1A", "label": "✅ 安心抱緊"},
    "review":       {"color": "#FFD600", "bg": "#2E2800", "label": "⚠️ 注意防線"},
    "partial_exit": {"color": "#FF9800", "bg": "#2E1500", "label": "🟡 分批停利"},
    "full_exit":    {"color": "#FF5252", "bg": "#2E0A0A", "label": "🔴 全數出場"},
}


def _alert_banner(text: str, text_color: str, bg_color: str) -> dict:
    return {
        "type": "box",
        "layout": "vertical",
        "backgroundColor": bg_color,
        "cornerRadius": "md",
        "paddingAll": "sm",
        "margin": "sm",
        "contents": [{
            "type": "text",
            "text": text,
            "color": text_color,
            "size": "sm",
            "weight": "bold",
            "align": "center",
            "wrap": True,
        }],
    }


def build_stock_report_flex(report: dict) -> dict:
    """
    接收 detect_swing_setup 回傳的 setup dict，回傳 LINE Flex Message payload。

    必要欄位：
        symbol, rating, setup_type, market_status,
        price, ema20, ema50, rsi, vol_ratio,
        entry_zone_low, entry_zone_high, planned_entry,
        stop_loss, stop_loss_price, trailing_stop,
        target_1, target_2, rr_ratio,
        exit_action, overheat_alert, storm_mode,
        distribution_alert, deviation_from_ma20_pct,
        news_summary (str)

    回傳值可直接塞入 LINE reply/push messages 陣列。
    """
    palette = _resolve_palette(report.get("rating", ""))
    symbol  = report.get("symbol", "N/A")
    price   = report.get("price", 0)

    # ── 警示橫幅（依優先序，最多顯示兩條）──────────────
    alert_banners: list[dict] = []
    if report.get("storm_mode"):
        alert_banners.append(_alert_banner(
            "⛈️ 風暴防禦模式 — 請抱現金休息",
            "#FF5252", "#1A0000",
        ))
    if report.get("overheat_alert"):
        alert_banners.append(_alert_banner(
            f"🔥 追高熔斷 — 正乖離 {report.get('deviation_from_ma20_pct', 0):.1f}%，等候拉回",
            "#FF9800", "#1A0A00",
        ))
    if report.get("distribution_alert"):
        alert_banners.append(_alert_banner(
            "🚨 大戶倒貨警報 — 建議全數出場",
            "#FF5252", "#1A0000",
        ))

    # ── Header ──────────────────────────────
    header = {
        "type": "box",
        "layout": "vertical",
        "backgroundColor": "#0D1117",
        "paddingAll": "lg",
        "contents": [
            # Symbol + 市場標籤
            {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {
                        "type": "text",
                        "text": symbol,
                        "color": "#FFFFFF",
                        "size": "xxl",
                        "weight": "bold",
                        "flex": 6,
                    },
                    {
                        "type": "text",
                        "text": report.get("market_status", ""),
                        "color": "#808080",
                        "size": "sm",
                        "align": "end",
                        "flex": 4,
                        "gravity": "bottom",
                    },
                ],
            },
            # 警示橫幅（若有）
            *alert_banners,
            # 評級 Badge
            {
                "type": "box",
                "layout": "vertical",
                "backgroundColor": palette["bg"],
                "cornerRadius": "md",
                "paddingAll": "sm",
                "margin": "md",
                "contents": [
                    {
                        "type": "text",
                        "text": palette["badge"],
                        "color": palette["text"],
                        "size": "md",
                        "weight": "bold",
                        "align": "center",
                    }
                ],
            },
            # Setup 類型
            {
                "type": "text",
                "text": f"Setup：{report.get('setup_type', 'N/A')}",
                "color": "#A0A0C0",
                "size": "sm",
                "margin": "sm",
                "align": "center",
            },
        ],
    }

    # ── Body ────────────────────────────────
    body_contents: list[dict] = []

    # 現價區塊
    body_contents += [
        _section_label("▸ 現價資訊"),
        _kv_row("現價", f"${price:,.2f}", "#FFFFFF"),
        _kv_row("EMA20", f"{report.get('ema20', 0):,.2f}"),
        _kv_row("EMA50", f"{report.get('ema50', 0):,.2f}"),
        _kv_row("RSI", f"{report.get('rsi', 0):.1f}"),
        _kv_row("量比", f"{report.get('vol_ratio', 0):.2f}x"),
        _divider(),
    ]

    # 進場計畫
    body_contents += [
        _section_label("▸ 進場計畫"),
        _kv_row("等待區", f"{report.get('entry_zone_low', 0):,.2f} – {report.get('entry_zone_high', 0):,.2f}"),
        _kv_row("計畫進場", f"{report.get('planned_entry', 0):,.2f}", "#FFD600"),
        _divider(),
    ]

    # 風控
    body_contents += [
        _section_label("▸ 風控設定"),
        _kv_row("ATR 止損",  f"{report.get('stop_loss', 0):,.2f}",  "#FF5252"),
        _kv_row("🛡️ 絕對止損", f"{report.get('stop_loss_price', 0):,.2f}", "#FF1744"),
        _kv_row("目標一",    f"{report.get('target_1', 0):,.2f}",    "#00E676"),
        _kv_row("目標二",    f"{report.get('target_2', 0):,.2f}",    "#00BFA5"),
        _kv_row("風報比",    f"{report.get('rr_ratio', 0):.2f} R",
                "#00E676" if report.get("rr_ratio", 0) >= 2.0 else "#FF5252"),
        _divider(),
    ]

    # 出場分析
    exit_action  = report.get("exit_action", "hold")
    exit_ep      = _EXIT_PALETTE.get(exit_action, _EXIT_PALETTE["hold"])
    trailing_stop = report.get("trailing_stop", 0)
    body_contents += [
        _section_label("▸ 持倉出場分析"),
        _kv_row("動態防守價", f"{trailing_stop:,.2f}", exit_ep["color"]),
        {
            "type": "box",
            "layout": "vertical",
            "backgroundColor": exit_ep["bg"],
            "cornerRadius": "md",
            "paddingAll": "sm",
            "margin": "sm",
            "contents": [{
                "type": "text",
                "text": exit_ep["label"],
                "color": exit_ep["color"],
                "size": "sm",
                "weight": "bold",
                "align": "center",
            }],
        },
        _divider(),
    ]

    # 新聞摘要
    news = report.get("news_summary", "")
    if news:
        body_contents += [
            _section_label("▸ 近期新聞"),
            {
                "type": "text",
                "text": news[:200],
                "color": "#909090",
                "size": "xs",
                "wrap": True,
                "margin": "sm",
            },
        ]

    body = {
        "type": "box",
        "layout": "vertical",
        "backgroundColor": "#0D1117",
        "paddingAll": "lg",
        "contents": body_contents,
    }

    # ── Footer ──────────────────────────────
    footer = {
        "type": "box",
        "layout": "vertical",
        "backgroundColor": "#111827",
        "paddingAll": "md",
        "contents": [
            {
                "type": "text",
                "text": "WengStock AI · 非投資建議 · 注意風險",
                "color": "#404060",
                "size": "xxs",
                "align": "center",
            }
        ],
    }

    # ── 組裝 Bubble ──────────────────────────
    bubble = {
        "type": "bubble",
        "size": "mega",
        "header": header,
        "body": body,
        "footer": footer,
        "styles": {
            "header": {"backgroundColor": "#0D1117"},
            "body":   {"backgroundColor": "#0D1117"},
            "footer": {"backgroundColor": "#111827"},
        },
    }

    return {
        "type": "flex",
        "altText": f"📊 {symbol} Swing 分析報告 — {palette['badge']}",
        "contents": bubble,
    }
