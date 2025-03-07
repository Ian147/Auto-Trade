import pandas as pd
import numpy as np
import time
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def get_binance_ohlcv(limit=100000, symbol="BTCUSDT", interval="15m"):
    """ Mengambil data OHLCV dari Binance """
    klines = client.get_historical_klines(symbol, interval, limit=limit)
    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# Ambil 100.000 data dan simpan
df = get_binance_ohlcv(100000)
df.to_csv("BTCUSDT_100k.csv")
print("✅ Data berhasil disimpan!")
