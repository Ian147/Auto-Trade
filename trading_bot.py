import ccxt
import time
import numpy as np
import requests
import pandas as pd
import threading
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Konfigurasi API Binance
api_key = "j70PupVRg6FbppOVsv0NJeyEYhf24fc9H36XvKQTP496CE8iQpuh0KlurfRGvrLw"
api_secret = "YGp4SiUdZMQ8ykAszgzSB1eLqv5ZiFM9wZTuV3Z2VOtoM46yDNuy1CBr703PtLVT"

# Konfigurasi API Telegram
telegram_token = "8011128170:AAEvCJrvMRinnIsInJmqLjzpWguz88tPWVw"
telegram_chat_id = "681125756"

# Inisialisasi Binance
binance = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"}
})

# Pair yang diperdagangkan
symbol = "BTC/USDT"
trade_amount = 10  # Order 10 USDT per transaksi
tp_percentage = 1.5 / 100  # TP +1.5%
sl_percentage = 1 / 100    # SL -1%

# Fungsi Kirim Notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Error mengirim pesan Telegram: {e}")

# Fungsi Open Order
def place_order(order_type):
    try:
        if order_type == "BUY":
            order = binance.create_market_buy_order(symbol, trade_amount / binance.fetch_ticker(symbol)["last"])
        else:
            order = binance.create_market_sell_order(symbol, trade_amount / binance.fetch_ticker(symbol)["last"])
        
        # Ambil harga eksekusi order terakhir
        entry_price = binance.fetch_my_trades(symbol)[-1]['price']

        send_telegram_message(f"📈 *{order_type} Order Executed*\n- Harga: {entry_price} USDT\n- TP: {entry_price * (1 + tp_percentage):.2f} USDT\n- SL: {entry_price * (1 - sl_percentage):.2f} USDT")

        return entry_price
    except Exception as e:
        send_telegram_message(f"⚠️ *Order Gagal:* {e}")
        return None

# Fungsi Cek TP dan SL (Menggunakan Threading)
def check_tp_sl(entry_price):
    def monitor_price():
        while True:
            ticker = binance.fetch_ticker(symbol)
            current_price = ticker['last']

            if current_price >= entry_price * (1 + tp_percentage):
                place_order("SELL")
                send_telegram_message(f"✅ *Take Profit Tercapai!* 🚀\n- Harga Jual: {current_price:.2f} USDT")
                break
            elif current_price <= entry_price * (1 - sl_percentage):
                place_order("SELL")
                send_telegram_message(f"⚠️ *Stop Loss Terpicu!* 📉\n- Harga Jual: {current_price:.2f} USDT")
                break

            time.sleep(5)  # Cek harga setiap 5 detik

    thread = threading.Thread(target=monitor_price)
    thread.start()

# Jalankan bot
def trading_bot():
    while True:
        try:
            current_price = binance.fetch_ticker(symbol)["last"]

            # Jika AI memberikan sinyal BUY
            entry_price = place_order("BUY")
            if entry_price:
                check_tp_sl(entry_price)

            time.sleep(60)  # Cek sinyal setiap 1 menit
        except Exception as e:
            send_telegram_message(f"⚠️ *Error:* {e}")
            time.sleep(10)

# Eksekusi bot
trading_bot()
