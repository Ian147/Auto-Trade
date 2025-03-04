import ccxt
import pandas as pd
import time
import os

# Inisialisasi Binance API
exchange = ccxt.binance()

# Pair yang ingin diambil
symbol = 'BTC/USDT'

# Timeframe
timeframe = '15m'

# Jumlah maksimum data OHLCV per request
limit = 1000

# Target total data
total_data = 1000000

# Cek apakah file sudah ada
file_name = 'data.csv'

if os.path.exists(file_name):
    df_old = pd.read_csv(file_name)
    
    # ✅ Ubah kolom timestamp ke datetime tanpa unit='ms'
    df_old['timestamp'] = pd.to_datetime(df_old['timestamp'])
    
    last_timestamp = df_old['timestamp'].max()
    since = int(last_timestamp.timestamp() * 1000) + 1  # Convert ke milidetik
    ohlcv_list = df_old.values.tolist()
    print(f"🔄 Melanjutkan pengambilan data dari {last_timestamp}...")
else:
    since = exchange.parse8601('2017-08-17T00:00:00Z')
    ohlcv_list = []

print("🚀 Mengunduh data dari Binance...")

while len(ohlcv_list) < total_data:
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe, since, limit)

        if not data:
            print("✅ Tidak ada data lagi.")
            break

        ohlcv_list.extend(data)
        since = data[-1][0] + 1

        print(f"📊 Data diunduh: {len(ohlcv_list)} / {total_data}")

        time.sleep(1)

    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(5)

# ✅ Konversi ke DataFrame dengan timestamp yang benar
df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

df.to_csv(file_name, index=False)
print(f"✅ Data berhasil disimpan sebagai '{file_name}' dengan {len(df)} baris.")
