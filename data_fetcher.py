import time
import pandas as pd
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET

# Inisialisasi Binance Client
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# Konfigurasi
PAIR = "BTCUSDT"  # Pasangan trading
TIMEFRAME = "15m"  # Timeframe data
TOTAL_DATA = 264070  # Target jumlah data
BATCH_SIZE = 1000  # Binance hanya mengizinkan max 1000 per request
OUTPUT_FILE = "binance_ohlcv_15m.csv"  # File penyimpanan data

def fetch_historical_data():
    """Mengunduh data OHLCV dari Binance secara bertahap."""
    all_data = []
    last_timestamp = None  # Untuk mengambil data mundur

    print(f"📥 Mengunduh {TOTAL_DATA} data OHLCV untuk {PAIR}...")
    
    while len(all_data) < TOTAL_DATA:
        try:
            # Ambil data dari Binance
            klines = client.get_klines(
                symbol=PAIR,
                interval=Client.KLINE_INTERVAL_15MINUTE,
                limit=BATCH_SIZE,
                endTime=last_timestamp
            )

            if not klines:
                print("✅ Tidak ada data lagi.")
                break

            # Format data ke DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            all_data.extend(df.values.tolist())

            # Perbarui timestamp terakhir agar mengambil data lebih lama
            last_timestamp = int(df.iloc[0]['timestamp'].timestamp() * 1000)

            # Tampilkan progres
            print(f"📊 Data diunduh: {len(all_data)} / {TOTAL_DATA}")

            # Hindari rate-limit Binance
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Error mengambil data: {e}")
            time.sleep(5)  # Coba lagi setelah 5 detik

    # Simpan data ke CSV
    df_final = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Data berhasil disimpan di {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_historical_data()
