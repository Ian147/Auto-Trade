import pandas as pd
import time
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET, PAIR, TIMEFRAME

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def get_binance_ohlcv(limit=1000):
    """ Mengambil data OHLCV dari Binance """
    klines = client.get_klines(symbol=PAIR, interval=TIMEFRAME, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', '_', '_', '_', '_', '_', '_'])
    
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    
    return df

# Simpan data ke CSV
df = get_binance_ohlcv(limit=264070)
df.to_csv("data.csv", index=True)
print("✅ Data berhasil disimpan: data.csv")
