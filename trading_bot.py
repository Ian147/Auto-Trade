import ccxt
import numpy as np
import pandas as pd
import time
import talib
import requests

# Konfigurasi API Binance
API_KEY = "j70PupVRg6FbppOVsv0NJeyEYhf24fc9H36XvKQTP496CE8iQpuh0KlurfRGvrLw"
API_SECRET = "YGp4SiUdZMQ8ykAszgzSB1eLqv5ZiFM9wZTuV3Z2VOtoM46yDNuy1CBr703PtLVT"

# Konfigurasi API Telegram
TELEGRAM_TOKEN = "8011128170:AAEvCJrvMRinnIsInJmqLjzpWguz88tPWVw"
TELEGRAM_CHAT_ID = "681125756"

# Inisialisasi Binance
binance = ccxt.binance({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "options": {"defaultType": "spot"},
    "enableRateLimit": True
})

# Fungsi kirim notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)

# Fungsi mengambil data pasar
def get_market_data(symbol, timeframe='1h', limit=100):
    candles = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['close'] = df['close'].astype(float)
    return df

# Fungsi analisis teknikal & prediksi sinyal trading
def get_trading_signal(symbol):
    df = get_market_data(symbol)

    # Indikator RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)

    # Indikator MACD
    df['macd'], df['signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Indikator Bollinger Bands
    df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['close'], timeperiod=20)

    # Strategi Trading:
    if df['rsi'].iloc[-1] < 30 and df['macd'].iloc[-1] > df['signal'].iloc[-1] and df['close'].iloc[-1] < df['lower'].iloc[-1]:
        return "BUY"
    elif df['rsi'].iloc[-1] > 70 and df['macd'].iloc[-1] < df['signal'].iloc[-1] and df['close'].iloc[-1] > df['upper'].iloc[-1]:
        return "SELL"
    else:
        return "HOLD"

# Fungsi untuk mengeksekusi order
def place_order(signal, symbol, balance_usdt):
    price = binance.fetch_ticker(symbol)['last']

    # Menentukan jumlah koin berdasarkan saldo USDT
    amount = balance_usdt / price

    # Take Profit & Stop Loss
    TP1 = price * 1.02   # Take Profit 1 (2% profit)
    TP2 = price * 1.05   # Take Profit 2 (5% profit)
    SL = price * 0.98    # Stop Loss (2% cut loss)

    if signal == "BUY":
        order = binance.create_market_buy_order(symbol, amount)
        send_telegram_message(f"📈 Open Buy {symbol} at {price}\n🎯 TP1: {TP1}\n🎯 TP2: {TP2}\n⛔ SL: {SL}")
        monitor_position(symbol, amount, TP1, TP2, SL)

    elif signal == "SELL":
        order = binance.create_market_sell_order(symbol, amount)
        send_telegram_message(f"📉 Open Sell {symbol} at {price}\n🎯 TP1: {TP1}\n🎯 TP2: {TP2}\n⛔ SL: {SL}")

# Fungsi untuk memantau posisi & eksekusi TP/SL
def monitor_position(symbol, amount, TP1, TP2, SL):
    while True:
        price = binance.fetch_ticker(symbol)['last']

        if price >= TP1:
            binance.create_market_sell_order(symbol, amount / 2)  # Jual 50% di TP1
            send_telegram_message(f"✅ TP1 Tercapai! {symbol} at {TP1}")

        if price >= TP2:
            binance.create_market_sell_order(symbol, amount / 2)  # Jual sisa 50% di TP2
            send_telegram_message(f"🚀 TP2 Tercapai! {symbol} at {TP2}")
            break  # Keluar dari loop setelah TP2 tercapai

        if price <= SL:
            binance.create_market_sell_order(symbol, amount)  # Cut Loss
            send_telegram_message(f"⚠️ Stop Loss Terpicu! {symbol} at {SL}")
            break  # Keluar dari loop setelah SL tercapai

        time.sleep(10)  # Cek harga setiap 10 detik

# Loop utama trading bot
symbol = "DOGE/USDT"  # Pasangan koin yang ditradingkan
balance = binance.fetch_balance()['USDT']['free']  # Cek saldo USDT

while True:
    signal = get_trading_signal(symbol)
    
    if signal != "HOLD":
        place_order(signal, symbol, balance)
    
    time.sleep(60)  # Cek setiap 1 menit
