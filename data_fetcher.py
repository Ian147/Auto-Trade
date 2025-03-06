import pandas as pd
import time
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET, PAIR, TIMEFRAME

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def get_binance_ohlcv(limit=260000):
    all_data = []
    last_timestamp = None

    while len(all_data) < limit:
        try:
            candles = client.get_klines(symbol=PAIR, interval=TIMEFRAME, limit=1000, startTime=last_timestamp)
            if not candles:
                break
            all_data.extend(candles)
            last_timestamp = candles[-1][0] + 1
            print(f"Downloaded: {len(all_data)} / {limit}")
            time.sleep(1)  # Hindari rate limit Binance
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(10)

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.to_csv("binance_ohlcv_15m.csv", index=False)
    print("✅ Data berhasil disimpan!")

if __name__ == "__main__":
    get_binance_ohlcv()
