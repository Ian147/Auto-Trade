import pandas as pd
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET, PAIR

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def get_binance_ohlcv(limit=100, interval=Client.KLINE_INTERVAL_15MINUTE):
    try:
        klines = client.get_klines(symbol=PAIR, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=["time", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
        df["close"] = df["close"].astype(float)
        return df
    except Exception as e:
        print(f"❌ Gagal mendapatkan data OHLCV: {e}")
        return None
