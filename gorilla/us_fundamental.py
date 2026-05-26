"""
大猩猩策略 — 美股基本面（SEC EDGAR XBRL API，完全免費）
抓取：Revenue、EPS、Gross Profit 季度數據，計算 YoY 成長率與 PEG
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import yfinance as yf

# SEC 要求 User-Agent 含聯絡資訊
_SEC_EMAIL    = os.getenv("SEC_USER_AGENT_EMAIL", "readinegogo@gmail.com")
SEC_HEADERS   = {"User-Agent": f"WengStockAI/1.0 {_SEC_EMAIL}"}
TICKERS_URL   = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL     = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_CACHE_FILE   = Path(__file__).parent.parent / ".sec_cik_cache.json"
_CACHE_TTL    = 86400 * 7  # 7 天更新一次 CIK 表

# 常見 GAAP 欄位（按優先順序，找到第一個有數據的就用）
_REVENUE_KEYS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "RevenuesNetOfInterestExpense",
    "SalesRevenueGoodsNet",
]
_EPS_KEYS = [
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
]
_GROSS_PROFIT_KEYS = ["GrossProfit"]


# ─────────────────────────────────────────────
# CIK 查詢（帶本地快取，避免每次下載 3MB 大表）
# ─────────────────────────────────────────────

def _load_cik_cache() -> dict:
    try:
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text())
            if time.time() - data.get("_ts", 0) < _CACHE_TTL:
                return data
    except Exception:
        pass
    return {}


def _save_cik_cache(cache: dict) -> None:
    cache["_ts"] = time.time()
    _CACHE_FILE.write_text(json.dumps(cache))


async def _get_cik(ticker: str, client: httpx.AsyncClient) -> int:
    cache = _load_cik_cache()
    key   = ticker.upper()
    if key in cache:
        return int(cache[key])

    resp = await client.get(TICKERS_URL, headers=SEC_HEADERS, timeout=30)
    data = resp.json()

    new_cache = {v["ticker"]: v["cik_str"] for v in data.values()}
    _save_cik_cache(new_cache)

    if key not in new_cache:
        raise ValueError(f"找不到 {ticker} 的 SEC CIK 編號")
    return int(new_cache[key])


# ─────────────────────────────────────────────
# XBRL 解析工具
# ─────────────────────────────────────────────

def _extract_quarterly(facts: dict, keys: list[str]) -> list[dict]:
    """
    從 XBRL companyfacts 抓指定欄位的季度數據。
    回傳依 end date 降序排列（最新季度在前）的 list。
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for key in keys:
        if key not in gaap:
            continue
        for unit_vals in gaap[key].get("units", {}).values():
            # 只要 10-Q / 10-K 且 fiscal period 是季度（Q1~Q4）
            quarterly = [
                v for v in unit_vals
                if v.get("form") in ("10-Q", "10-K")
                and str(v.get("fp", "")).startswith("Q")
            ]
            if not quarterly:
                continue
            # 同一期間保留最新申報版本
            seen: dict[str, dict] = {}
            for v in quarterly:
                end = v["end"]
                if end not in seen or v["filed"] > seen[end]["filed"]:
                    seen[end] = v
            result = sorted(seen.values(), key=lambda x: x["end"], reverse=True)
            if len(result) >= 4:
                return result
    return []


def _yoy(values: list[dict]) -> float | None:
    """最新季 vs 去年同季 YoY 成長率（%）"""
    if len(values) < 5:
        return None
    cur    = values[0]["val"]
    yr_ago = values[4]["val"]
    if yr_ago == 0:
        return None
    return (cur - yr_ago) / abs(yr_ago) * 100


def _ttm_growth(values: list[dict]) -> float | None:
    """最近四季 TTM vs 前四季 TTM 成長率，用於計算 PEG"""
    if len(values) < 8:
        return None
    ttm_cur  = sum(v["val"] for v in values[:4])
    ttm_prev = sum(v["val"] for v in values[4:8])
    if ttm_prev <= 0:
        return None
    return (ttm_cur - ttm_prev) / ttm_prev * 100


# ─────────────────────────────────────────────
# 主要對外函數
# ─────────────────────────────────────────────

async def get_us_fundamentals(ticker: str) -> dict:
    """
    從 SEC EDGAR 抓美股季報數據並計算大猩猩策略所需指標。
    Returns dict with keys:
        revenue_yoy, eps_yoy, gross_margin_current, gross_margin_change,
        eps_growth_rate (for PEG), peg, loss_to_profit, latest_quarter
        error (if failed)
    """
    async with httpx.AsyncClient(timeout=40) as client:
        try:
            cik  = await _get_cik(ticker, client)
        except ValueError as e:
            return {"error": str(e)}

        await _rate_limit()
        resp = await client.get(FACTS_URL.format(cik=cik), headers=SEC_HEADERS, timeout=40)
        if resp.status_code != 200:
            return {"error": f"SEC API 錯誤 {resp.status_code}"}
        facts = resp.json()

    rev_vals = _extract_quarterly(facts, _REVENUE_KEYS)
    eps_vals = _extract_quarterly(facts, _EPS_KEYS)
    gp_vals  = _extract_quarterly(facts, _GROSS_PROFIT_KEYS)

    revenue_yoy = _yoy(rev_vals)
    eps_yoy     = _yoy(eps_vals)

    # 虧轉盈偵測
    loss_to_profit = (
        len(eps_vals) >= 5
        and eps_vals[4]["val"] < 0
        and eps_vals[0]["val"] > 0
    )

    # 毛利率
    gross_margin_current = None
    gross_margin_change  = None
    if gp_vals and rev_vals and rev_vals[0]["val"] > 0:
        gross_margin_current = gp_vals[0]["val"] / rev_vals[0]["val"] * 100
        if len(gp_vals) >= 5 and len(rev_vals) >= 5 and rev_vals[4]["val"] > 0:
            gm_yago = gp_vals[4]["val"] / rev_vals[4]["val"] * 100
            gross_margin_change = gross_margin_current - gm_yago

    # EPS 成長率（TTM）→ PEG
    eps_growth_rate = _ttm_growth(eps_vals)
    peg             = _calc_peg(ticker, eps_growth_rate)

    return {
        "revenue_yoy":          revenue_yoy,
        "eps_yoy":              eps_yoy,
        "gross_margin_current": gross_margin_current,
        "gross_margin_change":  gross_margin_change,
        "eps_growth_rate":      eps_growth_rate,
        "peg":                  peg,
        "loss_to_profit":       loss_to_profit,
        "latest_quarter":       rev_vals[0]["end"] if rev_vals else None,
    }


def _calc_peg(ticker: str, eps_growth_rate: float | None) -> float | None:
    if not eps_growth_rate or eps_growth_rate <= 0:
        return None
    try:
        info = yf.Ticker(ticker).info
        pe   = info.get("trailingPE") or info.get("forwardPE")
        if not pe or pe <= 0:
            return None
        return round(pe / eps_growth_rate, 2)
    except Exception:
        return None


_last_sec_call = 0.0

async def _rate_limit() -> None:
    """SEC EDGAR 要求最多 10 req/s，加保守間隔"""
    global _last_sec_call
    import asyncio
    elapsed = time.time() - _last_sec_call
    if elapsed < 0.15:
        await asyncio.sleep(0.15 - elapsed)
    _last_sec_call = time.time()
