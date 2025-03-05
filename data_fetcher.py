import pandas as pd
import time
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET, PAIR, TIMEFRAME, DATA_PATH

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def get_historical_data(symbol, interval, limit=1000):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                       'close_time', 'quote_asset_volume', 'number_of_trades', 
                                       'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

data_list = []
while len(data_list) < 250000:
    data = get_historical_data(PAIR, TIMEFRAME)
    data_list.extend(data.values.tolist())
    print(f"📊 Data diunduh: {len(data_list)} / 250000")
    time.sleep(1)

df = pd.DataFrame(data_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df.to_csv(DATA_PATH, index=False)
print("✅ Data disimpan di", DATA_PATH)
