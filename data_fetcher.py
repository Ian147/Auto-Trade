import pandas as pd
import time
import logging
from binance.client import Client

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# API Binance (gunakan API sendiri)
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"

# Konfigurasi pasangan trading dan interval waktu
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
DATA_PATH = "data.csv"

# Koneksi ke Binance
client = Client(API_KEY, API_SECRET)

def fetch_historical_data():
    """Mengambil data historis OHLCV dari Binance"""
    logging.info(f"📥 Mengunduh data {SYMBOL} dengan interval {INTERVAL}...")

    # Mulai dari timestamp awal (misalnya 2017)
    start_str = "2017-08-17 04:00:00"
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)  # Ubah ke milidetik
    end_ts = int(pd.Timestamp.now().timestamp() * 1000)  # Timestamp sekarang
    
    all_data = []
    
    while start_ts < end_ts:
        logging.info(f"📊 Mengambil data dari {pd.to_datetime(start_ts, unit='ms')}...")
        
        # Ambil data dari Binance
        klines = client.get_klines(symbol=SYMBOL, interval=INTERVAL, startTime=start_ts, limit=1000)
        if not klines:
            break  # Jika tidak ada data lagi, berhenti
        
        for entry in klines:
            all_data.append([
                pd.to_datetime(entry[0], unit='ms'),  # Timestamp
                float(entry[1]),  # Open
                float(entry[2]),  # High
                float(entry[3]),  # Low
                float(entry[4]),  # Close
                float(entry[5])   # Volume
            ])
        
        start_ts = klines[-1][0] + 1  # Update timestamp ke data berikutnya
        time.sleep(1)  # Hindari rate limit

    # Simpan ke CSV
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.to_csv(DATA_PATH, index=False)
    logging.info(f"✅ Data disimpan ke {DATA_PATH} ({len(df)} baris)")

if __name__ == "__main__":
    fetch_historical_data()
