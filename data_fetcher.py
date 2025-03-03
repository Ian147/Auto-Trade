import pandas as pd
import numpy as np
import requests
import time
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET, PAIR, TIMEFRAME

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def get_binance_ohlcv(limit=1000):
    """ Mengambil data OHLCV dari Binance """
    klines = client.get_klines(symbol=PAIR, interval=TIMEFRAME, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                       'close_time', 'quote_asset_volume', 'trades', 
                                       'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

if __name__ == "__main__":
    data = get_binance_ohlcv(100)
    print(data.tail())
