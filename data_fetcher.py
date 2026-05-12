import yfinance as yf
import pandas as pd
import requests

# 建立一個 session 並加上 User-Agent 偽裝成瀏覽器
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

def get_market_intelligence(symbol):
    # 抓取最近 60 天的日 K 資料
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="60d")
    
    # 計算技術指標
    df['SMA20'] = ta.sma(df['Close'], length=20)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 格式化數據，這是餵給 AI 的「情報」
    intelligence = {
        "ticker": symbol,
        "price": round(latest['Close'], 2),
        "change_pct": round(((latest['Close'] - prev['Close']) / prev['Close']) * 100, 2),
        "rsi": round(latest['RSI'], 2),
        "trend": "Bullish" if latest['Close'] > latest['SMA20'] else "Bearish",
        "volume_surge": True if latest['Volume'] > df['Volume'].mean() * 1.5 else False
    }
    return intelligence

# 測試：看能不能抓到輝達 (NVDA) 的情報
if __name__ == "__main__":
    print(get_market_intelligence("NVDA"))