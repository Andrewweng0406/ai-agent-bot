"""
大猩猩策略 — 大盤風向球
美股：Nasdaq + S&P500 同時站上 50 SMA → 安全
台股：加權指數站上 60 MA → 安全
"""
from __future__ import annotations

import yfinance as yf


def check_market_safe() -> dict:
    """
    Returns:
        us_safe: bool
        tw_safe: bool
        us_detail: str   (說明原因)
        tw_detail: str
    """
    # ── 美股 ────────────────────────────────────
    us_safe    = True
    us_details = []

    for ticker, name in [("^IXIC", "Nasdaq"), ("^GSPC", "S&P500")]:
        try:
            data = yf.Ticker(ticker).history(period="4mo", interval="1d")
            if data.empty or len(data) < 50:
                continue
            price  = float(data["Close"].iloc[-1])
            sma50  = float(data["Close"].rolling(50).mean().iloc[-1])
            above  = price > sma50
            status = "✅" if above else "❌"
            us_details.append(f"{status} {name} {price:,.0f} {'>' if above else '<'} 50SMA {sma50:,.0f}")
            if not above:
                us_safe = False
        except Exception:
            continue

    # ── 台股 ────────────────────────────────────
    tw_safe    = True
    tw_details = []

    try:
        data = yf.Ticker("^TWII").history(period="6mo", interval="1d")
        if not data.empty and len(data) >= 60:
            price = float(data["Close"].iloc[-1])
            ma60  = float(data["Close"].rolling(60).mean().iloc[-1])
            above = price > ma60
            status = "✅" if above else "❌"
            tw_details.append(f"{status} 加權指數 {price:,.0f} {'>' if above else '<'} 60MA {ma60:,.0f}")
            if not above:
                tw_safe = False
    except Exception:
        pass

    return {
        "us_safe":    us_safe,
        "tw_safe":    tw_safe,
        "us_detail":  "\n".join(us_details) or "無法取得美股數據",
        "tw_detail":  "\n".join(tw_details) or "無法取得台股數據",
    }


def format_market_status() -> str:
    status = check_market_safe()
    lines  = ["🌐 大盤風向球\n"]

    us_label = "✅ 美股安全，可進場" if status["us_safe"] else "🚫 美股偏弱，暫停買進"
    tw_label = "✅ 台股安全，可進場" if status["tw_safe"] else "🚫 台股偏弱，暫停買進"

    lines.append(f"🇺🇸 {us_label}")
    lines.append(status["us_detail"])
    lines.append(f"\n🇹🇼 {tw_label}")
    lines.append(status["tw_detail"])

    return "\n".join(lines)
