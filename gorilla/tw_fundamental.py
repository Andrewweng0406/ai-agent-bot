"""
大猩猩策略 — 台股基本面（FinMind API，免費 600次/小時）
抓取：月營收、季報 EPS、三大法人籌碼
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import httpx

FINMIND_BASE  = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")


# ─────────────────────────────────────────────
# 底層 HTTP
# ─────────────────────────────────────────────

async def _get(dataset: str, data_id: str, start_date: str) -> list[dict]:
    params = {
        "dataset":    dataset,
        "data_id":    data_id,
        "start_date": start_date,
        "token":      FINMIND_TOKEN,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(FINMIND_BASE, params=params)
    body = resp.json()
    if body.get("status") != 200:
        return []
    return body.get("data", [])


def _start(days_ago: int) -> str:
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


# ─────────────────────────────────────────────
# 月營收 — TaiwanStockMonthRevenue
# ─────────────────────────────────────────────

async def get_tw_monthly_revenue(ticker: str) -> dict:
    """
    計算最新月份的月營收年增率（YoY %）。
    FinMind 回傳欄位：date / stock_id / revenue / revenue_month / revenue_year
    """
    rows = await _get("TaiwanStockMonthRevenue", ticker, _start(420))
    if not rows:
        return {"revenue_yoy": None, "error": "無月營收數據"}

    rows = sorted(rows, key=lambda x: x["date"], reverse=True)

    # 建立 {(year, month): revenue} 快速查詢
    rev_map: dict[tuple, int] = {}
    for r in rows:
        key = (int(r["revenue_year"]), int(r["revenue_month"]))
        rev_map[key] = int(r["revenue"])

    latest        = rows[0]
    latest_year   = int(latest["revenue_year"])
    latest_month  = int(latest["revenue_month"])
    latest_rev    = int(latest["revenue"])
    yago_rev      = rev_map.get((latest_year - 1, latest_month))

    revenue_yoy = None
    if yago_rev and yago_rev > 0:
        revenue_yoy = (latest_rev - yago_rev) / yago_rev * 100

    return {
        "revenue_yoy":    revenue_yoy,
        "latest_revenue": latest_rev,
        "latest_date":    latest["date"],
    }


# ─────────────────────────────────────────────
# 季 EPS — TaiwanStockFinancialStatements
# ─────────────────────────────────────────────

async def get_tw_quarterly_eps(ticker: str) -> dict:
    """
    計算最新季 EPS YoY 成長率。
    FinMind 回傳欄位：date / stock_id / type / value / origin_name
    """
    rows = await _get("TaiwanStockFinancialStatements", ticker, _start(600))
    eps_rows = sorted(
        [r for r in rows if r.get("type") == "EPS"],
        key=lambda x: x["date"],
        reverse=True,
    )

    if len(eps_rows) < 5:
        return {"eps_yoy": None}

    cur     = float(eps_rows[0]["value"])
    yr_ago  = float(eps_rows[4]["value"])

    loss_to_profit = yr_ago < 0 and cur > 0

    if yr_ago == 0:
        eps_yoy = None
    else:
        eps_yoy = (cur - yr_ago) / abs(yr_ago) * 100

    return {
        "eps_yoy":        eps_yoy,
        "current_eps":    cur,
        "loss_to_profit": loss_to_profit,
        "latest_date":    eps_rows[0]["date"],
    }


# ─────────────────────────────────────────────
# 三大法人 — TaiwanStockInstitutionalInvestors
# ─────────────────────────────────────────────

async def get_tw_institutional(ticker: str) -> dict:
    """
    統計外資（Foreign_Investor）連續買超天數。
    連買 ≥ 3 天 → 法人籌碼面確認。
    """
    rows = await _get("TaiwanStockInstitutionalInvestors", ticker, _start(30))

    foreign = sorted(
        [r for r in rows if r.get("name") == "Foreign_Investor"],
        key=lambda x: x["date"],
        reverse=True,
    )

    if not foreign:
        return {"foreign_consecutive_buy": 0, "foreign_net_latest": 0}

    consecutive = 0
    for row in foreign:
        net = int(row.get("buy", 0)) - int(row.get("sell", 0))
        if net > 0:
            consecutive += 1
        else:
            break

    net_latest = int(foreign[0].get("buy", 0)) - int(foreign[0].get("sell", 0))

    return {
        "foreign_consecutive_buy": consecutive,
        "foreign_net_latest":      net_latest,
    }
