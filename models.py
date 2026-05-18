"""
SQLAlchemy 資料庫模型 + FastAPI Watchlist 路由。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, String, UniqueConstraint, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from market_router import MarketRouter, Market

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/wengstock")

# ─────────────────────────────────────────────
# 資料庫連線
# ─────────────────────────────────────────────

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DB = Annotated[Session, Depends(get_db)]


# ─────────────────────────────────────────────
# 資料庫模型
# ─────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    line_user_id  = Column(String(64), unique=True, nullable=False, index=True)
    is_premium    = Column(Boolean, default=False, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)

    watchlist = relationship("Watchlist", back_populates="user",
                             cascade="all, delete-orphan")


class Watchlist(Base):
    __tablename__ = "watchlist"
    __table_args__ = (
        # 同一用戶不能重複追蹤同一支股票
        UniqueConstraint("user_id", "stock_code", name="uq_user_stock"),
    )

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    stock_code  = Column(String(16), nullable=False)
    market_type = Column(String(8), nullable=False)   # "TW" | "US"
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="watchlist")


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)


# ─────────────────────────────────────────────
# Pydantic 請求 / 回應模型
# ─────────────────────────────────────────────

class WatchlistAddRequest(BaseModel):
    line_user_id: str = Field(..., max_length=64, example="U1234567890abcdef")
    stock_code:   str = Field(..., max_length=16,  example="NVDA")


class WatchlistRemoveRequest(BaseModel):
    line_user_id: str = Field(..., max_length=64)
    stock_code:   str = Field(..., max_length=16)


class WatchlistItem(BaseModel):
    stock_code:  str
    market_type: str
    created_at:  datetime

    class Config:
        from_attributes = True


class WatchlistResponse(BaseModel):
    line_user_id: str
    is_premium:   bool
    watchlist:    list[WatchlistItem]


# ─────────────────────────────────────────────
# 內部 helper
# ─────────────────────────────────────────────

_router = MarketRouter()


def _get_or_create_user(db: Session, line_user_id: str) -> User:
    user = db.query(User).filter(User.line_user_id == line_user_id).first()
    if not user:
        user = User(line_user_id=line_user_id)
        db.add(user)
        db.flush()   # 取得 id，但還不 commit
    return user


def _classify_market(stock_code: str) -> str:
    try:
        info = _router.classify(stock_code)
        return info.market.value.replace("_MARKET", "")   # "TW" | "US"
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"無法識別股票代號格式：'{stock_code}'（請輸入 4 位數字或 1-5 位英文字母）",
        )


# ─────────────────────────────────────────────
# FastAPI 路由
# ─────────────────────────────────────────────

watchlist_router = APIRouter(prefix="/api/watchlist", tags=["Watchlist"])


@watchlist_router.post("/add", status_code=status.HTTP_201_CREATED)
def add_to_watchlist(body: WatchlistAddRequest, db: DB) -> dict:
    """
    加入自選股。
    - 自動判斷台股 / 美股（MarketRouter）。
    - 同一用戶重複加入同一股票回傳 409。
    """
    market_type = _classify_market(body.stock_code)
    stock_code  = body.stock_code.upper()

    try:
        user = _get_or_create_user(db, body.line_user_id)

        # 檢查是否已存在
        exists = (
            db.query(Watchlist)
            .filter(
                Watchlist.user_id   == user.id,
                Watchlist.stock_code == stock_code,
            )
            .first()
        )
        if exists:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"{stock_code} 已在您的自選股清單中。",
            )

        item = Watchlist(
            user_id     = user.id,
            stock_code  = stock_code,
            market_type = market_type,
        )
        db.add(item)
        db.commit()

        return {
            "message":    f"✅ {stock_code}（{market_type}）已加入自選股",
            "stock_code": stock_code,
            "market":     market_type,
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"資料庫錯誤：{str(e)}",
        )


@watchlist_router.post("/remove", status_code=status.HTTP_200_OK)
def remove_from_watchlist(body: WatchlistRemoveRequest, db: DB) -> dict:
    """
    移除自選股。找不到用戶或股票時回傳 404。
    """
    stock_code = body.stock_code.upper()

    try:
        user = db.query(User).filter(User.line_user_id == body.line_user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="找不到該用戶。",
            )

        item = (
            db.query(Watchlist)
            .filter(
                Watchlist.user_id    == user.id,
                Watchlist.stock_code == stock_code,
            )
            .first()
        )
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{stock_code} 不在您的自選股清單中。",
            )

        db.delete(item)
        db.commit()

        return {"message": f"🗑️ {stock_code} 已從自選股移除"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"資料庫錯誤：{str(e)}",
        )


@watchlist_router.get("/{line_user_id}", response_model=WatchlistResponse)
def get_watchlist(line_user_id: str, db: DB) -> WatchlistResponse:
    """查看某用戶的自選股清單。"""
    user = db.query(User).filter(User.line_user_id == line_user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="找不到該用戶。")

    return WatchlistResponse(
        line_user_id = user.line_user_id,
        is_premium   = user.is_premium,
        watchlist    = [
            WatchlistItem(
                stock_code  = w.stock_code,
                market_type = w.market_type,
                created_at  = w.created_at,
            )
            for w in user.watchlist
        ],
    )
