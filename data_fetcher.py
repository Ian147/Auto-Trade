import ccxt
import pandas as pd
import time

# Inisialisasi Binance API
exchange = ccxt.binance()

# Pair yang ingin diambil (BTC/USDT)
symbol = 'BTC/USDT'

# Timeframe 15m
timeframe = '15m'

# Jumlah maksimum data OHLCV per request (Binance limit = 1000)
limit = 1000

# Target total data (1.000.000 data)
total_data = 1000000

# Menyimpan semua data OHLCV
ohlcv_list = []

# Ambil timestamp awal (10 juta candle ke belakang dari sekarang)
since = exchange.parse8601('2017-08-17T00:00:00Z')

print("🚀 Mengunduh data dari Binance...")

while len(ohlcv_list) < total_data:
    # Ambil data dari Binance
    data = exchange.fetch_ohlcv(symbol, timeframe, since, limit)

    if not data:
        print("✅ Selesai mengunduh data.")
        break

    ohlcv_list.extend(data)

    # Update timestamp agar data tidak duplikat
    since = data[-1][0] + 1

    # Tampilkan progres
    print(f"📊 Data diunduh: {len(ohlcv_list)} / {total_data}")

    # Delay untuk menghindari rate limit
    time.sleep(1)

# Konversi ke DataFrame
df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Konversi timestamp ke format datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Simpan ke CSV
df.to_csv('data.csv', index=False)

print("✅ Data berhasil disimpan sebagai 'data.csv'")
