from binance.client import Client
import pandas as pd

# API Key Binance (ganti dengan milikmu)
api_key = "j70PupVRg6FbppOVsv0NJeyEYhf24fc9H36XvKQTP496CE8iQpuh0KlurfRGvrLw"
api_secret = "YGp4SiUdZMQ8ykAszgzSB1eLqv5ZiFM9wZTuV3Z2VOtoM46yDNuy1CBr703PtLVT"

# Koneksi ke Binance
client = Client(api_key, api_secret)

# Ambil data OHLCV untuk BTC/USDT (timeframe 15m)
klines = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_15MINUTE, limit=10000)

# Konversi ke DataFrame
df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
df = df[["timestamp", "open", "high", "low", "close", "volume"]]

# Simpan ke CSV
df.to_csv("ohlcv_data.csv", index=False)
print("Data OHLCV berhasil disimpan sebagai ohlcv_data.csv")
