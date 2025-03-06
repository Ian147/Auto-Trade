import time
import pandas as pd
import requests
from config import BINANCE_API_KEY, BINANCE_API_SECRET, PAIR, TIMEFRAME

# Konstanta
LIMIT = 1000  # Binance membatasi 1000 data per request
TOTAL_BARS = 260000  # Target jumlah data
BINANCE_URL = "https://api.binance.com/api/v3/klines"

def fetch_ohlcv(symbol, interval, limit, start_time=None):
    """Mengambil data OHLCV dari Binance."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_time:
        params["startTime"] = start_time

    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    
    try:
        response = requests.get(BINANCE_URL, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error mengambil data dari Binance: {e}")
        return None

def get_historical_data():
    """Mengambil 260.000 data OHLCV secara bertahap."""
    all_data = []
    start_time = None

    while len(all_data) < TOTAL_BARS:
        data = fetch_ohlcv(PAIR, TIMEFRAME, LIMIT, start_time)
        if not data:
            break  # Hentikan jika gagal mengambil data
        
        all_data.extend(data)
        start_time = data[-1][0]  # Mulai dari timestamp terakhir
        time.sleep(1)  # Hindari rate limit

        print(f"📊 Data diunduh: {len(all_data)} / {TOTAL_BARS}", end="\r")

    if len(all_data) < TOTAL_BARS:
        print("\n⚠️  Tidak cukup data tersedia dari Binance!")
    else:
        print("\n✅ Data berhasil diunduh!")

    return all_data[:TOTAL_BARS]  # Pastikan hanya 260.000 baris

def save_to_csv(data):
    """Simpan data OHLCV ke dalam CSV."""
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", 
                                     "close_time", "quote_asset_volume", "trades", 
                                     "taker_base_vol", "taker_quote_vol", "ignore"])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df.to_csv("data.csv", index=False)

    print(f"✅ Data disimpan ke data.csv ({len(df)} baris)")

if __name__ == "__main__":
    data = get_historical_data()
    if data:
        save_to_csv(data)
