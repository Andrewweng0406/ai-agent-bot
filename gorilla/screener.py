"""
大猩猩策略 — 核心篩選器（美股 + 台股）
每日收盤後自動掃描，找出符合 CAN SLIM 成長動能標準的標的。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import yfinance as yf

from .market_filter import check_market_safe
from .us_fundamental import get_us_fundamentals
from .tw_fundamental import get_tw_monthly_revenue, get_tw_quarterly_eps, get_tw_institutional

# ─────────────────────────────────────────────
# 掃描股票池
# ─────────────────────────────────────────────

US_UNIVERSE: list[str] = [
    # AI / 半導體
    "NVDA", "AMD", "AVGO", "ARM", "SMCI", "ANET", "MRVL", "LRCX", "KLAC", "ONTO",
    # 雲端 / SaaS 成長
    "CRWD", "PANW", "ZS", "NOW", "SNOW", "DDOG", "NET", "MNDY", "HUBS", "TTD",
    "APP", "AXON", "BILL", "GTLB", "DUOL",
    # 資料 / AI 平台
    "PLTR", "MDB", "ESTC", "AI",
    # 消費 / 電商成長
    "CELH", "SHOP", "MELI", "SE", "CHWY",
    # 金融科技
    "COIN", "AFRM", "NU", "SQ",
    # 太空 / 國防科技
    "RKLB", "ASTS", "IONQ",
    # 其他高成長
    "TSLA", "MSTR", "META", "NFLX", "UBER",
]

TW_UNIVERSE: list[str] = [
    # 半導體
    "2330", "2454", "2303", "3711", "6770", "2379", "3034", "2337",
    # AI 伺服器 / PCB
    "2395", "3231", "6669", "6414", "5483", "3037", "2383",
    # 電子製造
    "2317", "2308", "2382", "3008", "2357",
    # 網通 / IC 設計
    "3045", "6415", "2439", "3533",
    # 生技
    "4711", "6547", "1795",
    # 其他
    "2412", "2609",
]


# ─────────────────────────────────────────────
# 技術面工具（美股 + 台股共用）
# ─────────────────────────────────────────────

def _get_technical(ticker: str, is_tw: bool = False) -> dict | None:
    suffix = ".TW" if is_tw else ""
    try:
        hist = yf.Ticker(f"{ticker}{suffix}").history(period="14mo", interval="1d")
        if hist.empty or len(hist) < 50:
            return None

        close    = hist["Close"]
        volume   = hist["Volume"]
        price    = float(close.iloc[-1])
        sma50    = float(close.rolling(50).mean().iloc[-1])
        ma60     = float(close.rolling(60).mean().iloc[-1])
        sma200   = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        ma240    = float(close.rolling(240).mean().iloc[-1]) if len(close) >= 240 else None
        vol_avg  = float(volume.rolling(20).mean().iloc[-1])
        vol_now  = float(volume.iloc[-1])
        vol_ratio = vol_now / vol_avg if vol_avg > 0 else 0

        return {
            "price":     price,
            "sma50":     sma50,
            "sma200":    sma200,
            "ma60":      ma60,
            "ma240":     ma240,
            "vol_ratio": vol_ratio,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────
# 財報偵測（7 天內財報 → 警告，不直接 BUY）
# ─────────────────────────────────────────────

def _check_earnings(ticker: str) -> str | None:
    """
    回傳財報日期字串（如 "05/28"）若在未來 7 天內，否則回傳 None。
    只對美股做偵測，台股跳過（FinMind 財報日曆較難取得）。
    """
    try:
        t = yf.Ticker(ticker)

        # 優先用 earnings_dates（較準確）
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            now = datetime.now()
            for idx in ed.index:
                dt = idx.to_pydatetime().replace(tzinfo=None)
                days = (dt - now).days
                if -1 <= days <= 7:
                    return dt.strftime("%m/%d")

        # fallback: calendar
        cal = t.calendar
        if cal is not None and not cal.empty:
            date_col = next((c for c in cal.columns if "Earnings" in c), None)
            if date_col:
                for raw in cal[date_col].dropna():
                    dt = raw.to_pydatetime() if hasattr(raw, "to_pydatetime") else datetime.combine(raw, datetime.min.time())
                    dt = dt.replace(tzinfo=None)
                    days = (dt - datetime.now()).days
                    if -1 <= days <= 7:
                        return dt.strftime("%m/%d")
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# 美股大猩猩篩選
# ─────────────────────────────────────────────

async def screen_us(ticker: str) -> dict:
    """完整美股大猩猩篩選，回傳診斷結果 dict。"""
    tech = _get_technical(ticker)
    if not tech:
        return {"pass": False, "ticker": ticker, "reason": "技術數據不足"}

    fund = await get_us_fundamentals(ticker)
    if "error" in fund:
        return {"pass": False, "ticker": ticker, "reason": fund["error"]}

    passes: list[str] = []
    fails:  list[str] = []

    # ① 營收 YoY > 25%
    rev = fund.get("revenue_yoy")
    if rev is not None:
        (passes if rev >= 25 else fails).append(
            f"營收年增 {rev:.1f}%{'✅' if rev >= 25 else '（需>25%）'}"
        )
    else:
        fails.append("無法取得營收數據")

    # ② EPS YoY > 20% 或虧轉盈
    eps = fund.get("eps_yoy")
    l2p = fund.get("loss_to_profit", False)
    if l2p:
        passes.append("EPS 虧轉盈 ✅")
    elif eps is not None:
        (passes if eps >= 20 else fails).append(
            f"EPS 年增 {eps:.1f}%{'✅' if eps >= 20 else '（需>20%）'}"
        )
    else:
        fails.append("無法取得 EPS 數據")

    # ③ 毛利率改善
    gm  = fund.get("gross_margin_current")
    gmc = fund.get("gross_margin_change")
    if gmc is not None:
        (passes if gmc >= 0 else fails).append(
            f"毛利率 {gm:.1f}%（YoY {gmc:+.1f}%）{'✅' if gmc >= 0 else '↓'}"
        )

    # ④ 成交量 > 1.5x
    vr = tech["vol_ratio"]
    (passes if vr >= 1.5 else fails).append(
        f"成交量 {vr:.1f}x{'✅' if vr >= 1.5 else '（需>1.5x）'}"
    )

    # ⑤ PEG < 1.2
    peg = fund.get("peg")
    if peg is not None:
        (passes if peg < 1.2 else fails).append(
            f"PEG {peg:.2f}{'✅' if peg < 1.2 else '（偏高）'}"
        )

    # ⑥ 技術多頭排列
    pr, s50, s200 = tech["price"], tech["sma50"], tech["sma200"]
    (passes if pr > s50 else fails).append(
        f"收盤 {'>' if pr > s50 else '<'} 50SMA"
    )
    if s200:
        (passes if s50 > s200 else fails).append(
            f"50SMA {'>' if s50 > s200 else '<'} 200SMA"
        )

    passed = len(fails) == 0
    earnings_warning = _check_earnings(ticker) if passed else None

    return {
        "ticker":               ticker,
        "is_tw":                False,
        "pass":                 passed,
        "passes":               passes,
        "fails":                fails,
        "score":                len(passes),
        "revenue_yoy":          rev,
        "eps_yoy":              eps,
        "loss_to_profit":       l2p,
        "gross_margin_current": gm,
        "gross_margin_change":  gmc,
        "peg":                  peg,
        "price":                tech["price"],
        "sma50":                tech["sma50"],
        "sma200":               tech["sma200"],
        "vol_ratio":            tech["vol_ratio"],
        "latest_quarter":       fund.get("latest_quarter"),
        "earnings_warning":     earnings_warning,
    }


# ─────────────────────────────────────────────
# 台股大猩猩篩選
# ─────────────────────────────────────────────

async def screen_tw(ticker: str) -> dict:
    """完整台股大猩猩篩選。"""
    tech = _get_technical(ticker, is_tw=True)
    if not tech:
        return {"pass": False, "ticker": ticker, "reason": "技術數據不足"}

    rev_data  = await get_tw_monthly_revenue(ticker)
    eps_data  = await get_tw_quarterly_eps(ticker)
    inst_data = await get_tw_institutional(ticker)

    passes: list[str] = []
    fails:  list[str] = []

    # ① 月營收 YoY > 20%
    rev = rev_data.get("revenue_yoy")
    if rev is not None:
        (passes if rev >= 20 else fails).append(
            f"月營收年增 {rev:.1f}%{'✅' if rev >= 20 else '（需>20%）'}"
        )
    else:
        fails.append("無法取得月營收")

    # ② EPS YoY > 20% 或虧轉盈
    eps = eps_data.get("eps_yoy")
    l2p = eps_data.get("loss_to_profit", False)
    if l2p:
        passes.append("EPS 虧轉盈 ✅")
    elif eps is not None:
        (passes if eps >= 20 else fails).append(
            f"季 EPS 年增 {eps:.1f}%{'✅' if eps >= 20 else '（需>20%）'}"
        )

    # ③ 外資連買 ≥ 3 天
    consec = inst_data.get("foreign_consecutive_buy", 0)
    (passes if consec >= 3 else fails).append(
        f"外資連買 {consec} 天{'✅' if consec >= 3 else '（需≥3天）'}"
    )

    # ④ 成交量 > 1.5x
    vr = tech["vol_ratio"]
    (passes if vr >= 1.5 else fails).append(
        f"成交量 {vr:.1f}x{'✅' if vr >= 1.5 else '（需>1.5x）'}"
    )

    # ⑤ 技術多頭排列（台股用 60MA / 240MA）
    pr, m60, m240 = tech["price"], tech["ma60"], tech["ma240"]
    (passes if pr > m60 else fails).append(
        f"收盤 {'>' if pr > m60 else '<'} 60MA"
    )
    if m240:
        (passes if m60 > m240 else fails).append(
            f"60MA {'>' if m60 > m240 else '<'} 240MA"
        )

    passed = len(fails) == 0

    return {
        "ticker":            ticker,
        "is_tw":             True,
        "pass":              passed,
        "passes":            passes,
        "fails":             fails,
        "score":             len(passes),
        "revenue_yoy":       rev,
        "eps_yoy":           eps,
        "loss_to_profit":    l2p,
        "consecutive_buy":   consec,
        "price":             tech["price"],
        "ma60":              tech["ma60"],
        "ma240":             tech["ma240"],
        "vol_ratio":         tech["vol_ratio"],
    }


# ─────────────────────────────────────────────
# 每日全市場掃描
# ─────────────────────────────────────────────

async def run_daily_scan(market: str) -> dict:
    """
    market: "US" or "TW"
    Returns:
        market_safe: bool
        market_detail: str
        picks: list[dict]   (通過篩選的標的，依分數排序)
    """
    mkt = check_market_safe()

    if market == "US":
        safe   = mkt["us_safe"]
        detail = mkt["us_detail"]
    else:
        safe   = mkt["tw_safe"]
        detail = mkt["tw_detail"]

    if not safe:
        return {"market_safe": False, "market_detail": detail, "picks": []}

    universe = US_UNIVERSE if market == "US" else TW_UNIVERSE
    tasks    = [screen_us(t) if market == "US" else screen_tw(t) for t in universe]

    # 限制並發，避免衝擊 SEC / FinMind rate limit
    results = []
    sem     = asyncio.Semaphore(5)

    async def _guarded(coro):
        async with sem:
            try:
                return await coro
            except Exception:
                return None

    gathered = await asyncio.gather(*[_guarded(t) for t in tasks])
    picks    = [r for r in gathered if r and r.get("pass")]
    picks.sort(key=lambda x: x.get("score", 0), reverse=True)

    return {
        "market_safe":   True,
        "market_detail": detail,
        "picks":         picks,
    }
