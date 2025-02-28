import ccxt
import pandas as pd
import numpy as np
import talib
from config import API_KEY, API_SECRET, SYMBOL, TIMEFRAME

# Inisialisasi Binance
binance = ccxt.binance({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "options": {"defaultType": "spot"}
})

# Ambil data OHLCV dari Binance
def get_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=100):
    bars = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# Hitung indikator teknikal
def calculate_indicators(df):
    df["RSI"] = talib.RSI(df["close"], timeperiod=14)
    
    macd, macdsignal, _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_Signal"] = macdsignal
    
    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20)
    df["Upper_BB"] = upper
    df["Middle_BB"] = middle
    df["Lower_BB"] = lower

    return df

# Simpan data ke CSV
def save_data(df, filename="market_data.csv"):
    df.to_csv(filename, index=False)
    print(f"✅ Data {SYMBOL} disimpan ke {filename}")
