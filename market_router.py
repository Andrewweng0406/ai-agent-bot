import re
import yfinance as yf
import pandas as pd

from enum import Enum
from typing import Protocol, runtime_checkable
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# 市場類型
# ─────────────────────────────────────────────

class Market(str, Enum):
    TW = "TW_MARKET"
    US = "US_MARKET"


# ─────────────────────────────────────────────
# 資料模型
# ─────────────────────────────────────────────

class SymbolInfo(BaseModel):
    raw: str = Field(..., description="使用者原始輸入")
    symbol: str = Field(..., description="標準化後的代號（yfinance 格式）")
    market: Market

class FetchResult(BaseModel):
    symbol: str
    market: Market
    daily: object      # pd.DataFrame，Pydantic 不驗證 DataFrame
    h4: object
    h1: object

    class Config:
        arbitrary_types_allowed = True


# ─────────────────────────────────────────────
# 資料服務 Protocol（依賴反轉，方便 Mock 測試）
# ─────────────────────────────────────────────

@runtime_checkable
class DataService(Protocol):
    def fetch(self, symbol: str) -> FetchResult:
        ...


# ─────────────────────────────────────────────
# 台股資料服務
# ─────────────────────────────────────────────

class TWDataService:
    def fetch(self, symbol: str) -> FetchResult:
        yf_symbol = f"{symbol}.TW"
        ticker = yf.Ticker(yf_symbol)

        daily = ticker.history(period="6mo", interval="1d", auto_adjust=False)
        h4 = ticker.history(period="60d", interval="1h", auto_adjust=False)
        h1 = ticker.history(period="30d", interval="1h", auto_adjust=False)

        # 上市找不到時，改找上櫃
        if daily.empty:
            yf_symbol = f"{symbol}.TWO"
            ticker = yf.Ticker(yf_symbol)
            daily = ticker.history(period="6mo", interval="1d", auto_adjust=False)
            h4 = ticker.history(period="60d", interval="1h", auto_adjust=False)
            h1 = ticker.history(period="30d", interval="1h", auto_adjust=False)

        return FetchResult(
            symbol=yf_symbol,
            market=Market.TW,
            daily=daily,
            h4=h4,
            h1=h1,
        )


# ─────────────────────────────────────────────
# 美股資料服務
# ─────────────────────────────────────────────

class USDataService:
    def fetch(self, symbol: str) -> FetchResult:
        ticker = yf.Ticker(symbol)

        daily = ticker.history(period="6mo", interval="1d", auto_adjust=False)
        h4 = ticker.history(period="60d", interval="1h", auto_adjust=False)
        h1 = ticker.history(period="30d", interval="1h", auto_adjust=False)

        return FetchResult(
            symbol=symbol,
            market=Market.US,
            daily=daily,
            h4=h4,
            h1=h1,
        )


# ─────────────────────────────────────────────
# Market Router（核心分流器）
# ─────────────────────────────────────────────

class MarketRouter:
    _TW_PATTERN = re.compile(r"^\d{4}$")
    _US_PATTERN = re.compile(r"^[A-Za-z]{1,5}$")

    def __init__(
        self,
        tw_service: DataService | None = None,
        us_service: DataService | None = None,
    ) -> None:
        self._services: dict[Market, DataService] = {
            Market.TW: tw_service or TWDataService(),
            Market.US: us_service or USDataService(),
        }

    def classify(self, user_input: str) -> SymbolInfo:
        """
        判斷使用者輸入是台股還是美股。
        台股：4 位純數字（2330）
        美股：1-5 位純英文（NVDA）
        """
        raw = user_input.strip()
        normalized = raw.upper()

        if self._TW_PATTERN.match(raw):
            return SymbolInfo(raw=raw, symbol=raw, market=Market.TW)

        if self._US_PATTERN.match(raw):
            return SymbolInfo(raw=raw, symbol=normalized, market=Market.US)

        raise ValueError(
            f"無法識別代號格式：'{raw}'。"
            "請輸入 4 位數字（台股）或 1-5 位英文字母（美股）。"
        )

    def route(self, user_input: str) -> FetchResult:
        """
        分類後，自動呼叫對應市場的資料服務。
        """
        info = self.classify(user_input)
        service = self._services[info.market]
        return service.fetch(info.symbol)


# ─────────────────────────────────────────────
# FastAPI 整合層（可單獨掛到 app.py）
# ─────────────────────────────────────────────

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel as PydanticBase

app = FastAPI(title="WengStock Market Router", version="1.0.0")

class RouteRequest(PydanticBase):
    symbol: str = Field(..., example="NVDA", description="股票代號")

class RouteResponse(PydanticBase):
    symbol: str
    market: Market
    has_data: bool
    rows_daily: int


def get_router() -> MarketRouter:
    return MarketRouter()


@app.post("/route", response_model=RouteResponse, summary="分流並抓取股票資料")
def route_symbol(
    body: RouteRequest,
    router: MarketRouter = Depends(get_router),
) -> RouteResponse:
    try:
        result = router.route(body.symbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"資料抓取失敗：{str(e)}")

    return RouteResponse(
        symbol=result.symbol,
        market=result.market,
        has_data=not result.daily.empty,
        rows_daily=len(result.daily),
    )


@app.get("/classify/{symbol}", summary="只做分類，不抓資料")
def classify_only(
    symbol: str,
    router: MarketRouter = Depends(get_router),
) -> dict:
    try:
        info = router.classify(symbol)
        return info.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
