import ccxt
import pandas as pd
import numpy as np
import talib
import time

# Masukkan API Key Binance
api_key = "6ipqniXiFRmjwGsB8H9vUpgVTexAnsLZ2Ybi0DrLxSKKINMr42wCC8ex7rIrqNlj"
api_secret = "HeINMThVDiJuCaoZFvC16FNj0ZCx9uGs2BxkkS1qTB3PkGTmibXfba3l8DajJ3x0"

# Inisialisasi Binance
binance = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"}
})

# Fungsi untuk mengambil data candlestick
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

# Fungsi untuk mengeksekusi order beli
def buy(symbol="BTC/USDT", amount=0.001):
    order = binance.create_market_buy_order(symbol, amount)
    print(f"Order beli berhasil: {order}")

# Fungsi untuk mengeksekusi order jual
def sell(symbol="BTC/USDT", amount=0.001):
    order = binance.create_market_sell_order(symbol, amount)
    print(f"Order jual berhasil: {order}")

# Fungsi strategi trading otomatis
def trade_strategy(symbol="BTC/USDT"):
    df = get_ohlcv(symbol)
    df = calculate_indicators(df)

    latest = df.iloc[-1]

    print(f"Harga saat ini: {latest['close']}, RSI: {latest['rsi']}, MACD: {latest['macd']}, MACD Signal: {latest['macd_signal']}")

    # Sinyal beli jika MACD lebih besar dari MACD Signal dan RSI lebih rendah dari 30
    if latest["macd"] > latest["macd_signal"] and latest["rsi"] < 30:
        print("==> SINYAL BELI TERDETEKSI")
        buy(symbol)
    
    # Sinyal jual jika MACD lebih kecil dari MACD Signal dan RSI lebih besar dari 70
    elif latest["macd"] < latest["macd_signal"] and latest["rsi"] > 70:
        print("==> SINYAL JUAL TERDETEKSI")
        sell(symbol)
    
    # Tidak ada sinyal trading jika tidak memenuhi kondisi
    else:
        print("==> Tidak ada sinyal trading saat ini")

# Fungsi untuk mengambil data dan menyimpan ke CSV
def save_data_to_csv(symbol="BTC/USDT", filename="data.csv", timeframe="1h", limit=100):
    df = get_ohlcv(symbol, timeframe, limit)
    df.to_csv(filename, index=False)
    print(f"Data {symbol} berhasil disimpan ke {filename}.")

# Loop utama untuk menjalankan strategi secara otomatis
def main():
    symbol = "BTC/USDT"  # Ganti dengan simbol yang diinginkan
    filename = f"{symbol.replace('/', '_')}_data.csv"

    # Ambil dan simpan data pasar terkini
    save_data_to_csv(symbol, filename)

    while True:
        print("\nMenjalankan strategi trading...")
        trade_strategy(symbol)
        time.sleep(3600)  # Tunggu 1 jam untuk iterasi berikutnya

# Jalankan bot trading
if __name__ == "__main__":
    main()
