import requests
import pandas as pd
import time
import threading
from datetime import datetime, timedelta

# === Konfigurasi ===
API_URL = "https://api.binance.com/api/v3/klines"
PAIR = "BTCUSDT"
TIMEFRAME = "15m"
LIMIT = 1000  # Maksimum data per request (API Spot limit)
TOTAL_CANDLES = 260000  # Target total data yang ingin diambil
THREADS = 5  # Jumlah thread untuk mempercepat proses

# Hitung interval waktu berdasarkan timeframe
INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
    "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400
}[TIMEFRAME]

# Hitung total waktu yang perlu diambil
total_seconds = TOTAL_CANDLES * INTERVAL_SECONDS
end_time = int(time.time() * 1000)  # Waktu sekarang dalam milidetik
start_time = end_time - (total_seconds * 1000)

# === Fungsi untuk Fetch Data ===
def fetch_ohlcv(start, end):
    """Mengambil data OHLCV dari Binance dalam rentang waktu tertentu."""
    url = f"{API_URL}?symbol={PAIR}&interval={TIMEFRAME}&startTime={start}&endTime={end}&limit={LIMIT}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"⚠️ Gagal mengambil data: {response.text}")
        return []

# === Fungsi untuk Mendownload Data Secara Paralel ===
def download_data(start, end, results):
    """Fungsi untuk mendownload data dalam thread."""
    data = fetch_ohlcv(start, end)
    results.extend(data)
    print(f"✅ Download {len(data)} data dari {datetime.utcfromtimestamp(start//1000)} ke {datetime.utcfromtimestamp(end//1000)}")

# === Multi-threading Download ===
all_data = []
current_start = start_time
threads = []

while current_start < end_time:
    current_end = min(current_start + (LIMIT * INTERVAL_SECONDS * 1000), end_time)
    
    # Buat thread untuk mengambil data dalam rentang waktu tertentu
    thread = threading.Thread(target=download_data, args=(current_start, current_end, all_data))
    threads.append(thread)
    thread.start()
    
    # Update start time untuk request berikutnya
    current_start = current_end
    
    # Batasi jumlah thread yang berjalan bersamaan
    if len(threads) >= THREADS:
        for t in threads:
            t.join()
        threads = []

# Tunggu semua thread selesai
for t in threads:
    t.join()

# === Simpan ke CSV ===
df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trades", "taker_base_vol", "taker_quote_vol", "ignore"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
df = df[["timestamp", "open", "high", "low", "close", "volume"]]
df.to_csv("binance_ohlcv_15m.csv", index=False)

print(f"✅ Data berhasil disimpan! Total data: {len(df)}")
