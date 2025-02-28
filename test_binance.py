import ccxt
import pandas as pd
import numpy as np
import talib

# Masukkan API Key Binance
api_key = "6ipqniXiFRmjwGsB8H9vUpgVTexAnsLZ2Ybi0DrLxSKKINMr42wCC8ex7rIrqNlj"
api_secret = "HeINMThVDiJuCaoZFvC16FNj0ZCx9uGs2BxkkS1qTB3PkGTmibXfba3l8DajJ3x0"

# Inisialisasi Binance
binance = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"}
})

# Ambil data historis dari Binance
def get_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100):
    bars = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# Hitung indikator teknikal
def calculate_indicators(df):
    df["rsi"] = talib.RSI(df["close"], timeperiod=14)

    macd, macdsignal, macdhist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd
    df["macd_signal"] = macdsignal

    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df["upper_bb"] = upper
    df["middle_bb"] = middle
    df["lower_bb"] = lower

    return df

# Order beli
def buy(symbol="BTC/USDT", amount=0.001):
    print(f"Simulasi Order BELI {symbol} sejumlah {amount}")

# Order jual
def sell(symbol="BTC/USDT", amount=0.001):
    print(f"Simulasi Order JUAL {symbol} sejumlah {amount}")

# Strategi trading otomatis
def trade_strategy(symbol="BTC/USDT"):
    df = get_ohlcv(symbol)
    df = calculate_indicators(df)

    latest = df.iloc[-1]

    print(f"Harga saat ini: {latest['close']}, RSI: {latest['rsi']}, MACD: {latest['macd']}, MACD Signal: {latest['macd_signal']}")

    if latest["macd"] > latest["macd_signal"] and latest["rsi"] < 30:
        print("==> SINYAL BELI TERDETEKSI (Simulasi)")
        buy(symbol)
    elif latest["macd"] < latest["macd_signal"] and latest["rsi"] > 70:
        print("==> SINYAL JUAL TERDETEKSI (Simulasi)")
        sell(symbol)
    else:
        print("==> Tidak ada sinyal trading saat ini")

# Jalankan bot
trade_strategy()
