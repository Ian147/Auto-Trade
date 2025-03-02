import logging
import ccxt
import time
import requests
import pandas as pd
import threading
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Konfigurasi Logging
logging.basicConfig(filename='trading_bot.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
trade_amount = 0.0005 # Order 0.0005 BTC per transaksi
tp_percentage = 1.5 / 100  # TP +1.5%
sl_percentage = 1 / 100    # SL -1%

# Fungsi Kirim Notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        logging.error(f"Error mengirim pesan Telegram: {e}")

# Fungsi untuk menghitung akurasi sinyal
def calculate_accuracy(entry_price, predicted_direction):
    """
    Fungsi ini menghitung akurasi berdasarkan apakah harga bergerak sesuai dengan prediksi sinyal.
    """
    current_price = binance.fetch_ticker(symbol)["last"]
    
    # Jika prediksi "BUY" dan harga naik, maka sinyal benar (akurasi 100%)
    if predicted_direction == "BUY" and current_price > entry_price:
        accuracy = 100
    # Jika prediksi "SELL" dan harga turun, maka sinyal benar (akurasi 100%)
    elif predicted_direction == "SELL" and current_price < entry_price:
        accuracy = 100
    else:
        accuracy = 0  # Prediksi salah

    return accuracy

# Fungsi untuk mengambil saldo spot
def get_spot_balance():
    try:
        balance = binance.fetch_balance()
        usdt_balance = balance['total'].get('USDT', 0)  # Ambil saldo USDT di spot
        btc_balance = balance['total'].get('BTC', 0)    # Ambil saldo BTC di spot
        return usdt_balance, btc_balance
    except Exception as e:
        logging.error(f"Error mengambil saldo spot: {e}")
        return 0, 0

# Fungsi Open Order
def place_order(order_type):
    try:
        logging.info(f"Mencoba untuk membuka order {order_type}")
        if order_type == "BUY":
            order = binance.create_market_buy_order(symbol, trade_amount / binance.fetch_ticker(symbol)["last"])
        else:
            order = binance.create_market_sell_order(symbol, trade_amount / binance.fetch_ticker(symbol)["last"])
        
        # Ambil harga eksekusi order terakhir
        entry_price = binance.fetch_my_trades(symbol)[-1]['price']

        # Mengambil saldo spot
        usdt_balance, btc_balance = get_spot_balance()

        send_telegram_message(f"📈 *{order_type} Order Executed*\n- Harga: {entry_price} USDT\n- TP: {entry_price * (1 + tp_percentage):.2f} USDT\n- SL: {entry_price * (1 - sl_percentage):.2f} USDT\n"
                              f"💰 *Saldo Spot:* {usdt_balance:.2f} USDT, {btc_balance:.6f} BTC")

        # Menghitung akurasi sinyal
        accuracy = calculate_accuracy(entry_price, order_type)
        send_telegram_message(f"🔍 *Akurasi Sinyal:* {accuracy}%")
        
        logging.info(f"Order {order_type} berhasil dieksekusi pada harga {entry_price} USDT dengan akurasi sinyal {accuracy}%")
        return entry_price, accuracy
    except Exception as e:
        logging.error(f"Order {order_type} gagal: {e}")
        send_telegram_message(f"⚠️ *Order Gagal:* {e}")
        return None, 0

# Fungsi Cek TP dan SL (Menggunakan Threading)
def check_tp_sl(entry_price):
    def monitor_price():
        while True:
            ticker = binance.fetch_ticker(symbol)
            current_price = ticker['last']

            logging.info(f"Memeriksa harga: {current_price} USDT")

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
            logging.info(f"Harga saat ini: {current_price} USDT")

            # Jika AI memberikan sinyal BUY
            entry_price, accuracy = place_order("BUY")
            if entry_price:
                check_tp_sl(entry_price)

            time.sleep(60)  # Cek sinyal setiap 1 menit
        except Exception as e:
            logging.error(f"Error utama: {e}")
            send_telegram_message(f"⚠️ *Error:* {e}")
            time.sleep(10)

# Eksekusi bot
trading_bot()
