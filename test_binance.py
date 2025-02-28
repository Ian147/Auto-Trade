import ccxt
import pandas as pd
import talib

# API Key Binance (Gunakan akun uji coba atau atur ke None jika hanya ingin menguji indikator)
api_key = None
api_secret = None

# Inisialisasi Binance tanpa API key jika hanya ingin membaca data publik
binance = ccxt.binance() if api_key is None else ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"}  # Spot trading
})

# Fungsi untuk mengambil data harga
def get_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100):
    bars = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# Fungsi untuk menghitung indikator
def calculate_indicators(df):
    df["rsi"] = talib.RSI(df["close"], timeperiod=14)
    df["macd"], df["macd_signal"], _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["upper_bb"], df["middle_bb"], df["lower_bb"] = talib.BBANDS(df["close"], timeperiod=20)
    return df

# Jalankan pengujian
if __name__ == "__main__":
    df = get_ohlcv("BTC/USDT", "1h", 100)  # Ambil 100 candlestick terakhir
    df = calculate_indicators(df)
    print(df.tail())  # Tampilkan 5 data terakhir dengan indikator
