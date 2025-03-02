import logging
import ccxt
import time
import numpy as np
import requests
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import threading

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
trade_amount = 10  # Order 10 USDT per transaksi
tp_percentage = 5 / 100  # TP +5%
sl_percentage = 5 / 100  # SL -5%

# Fungsi Kirim Notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error mengirim pesan Telegram: {e}")

# Fungsi Mengecek Saldo Spot
def check_balance():
    try:
        balance = binance.fetch_balance()
        spot_balance = balance['total']['USDT']
        logging.info(f"Saldo spot: {spot_balance} USDT")
        return spot_balance
    except Exception as e:
        logging.error(f"Error saat mengecek saldo: {e}")
        return 0

# Fungsi Open Limit Order
def place_order(order_type):
    try:
        logging.info(f"Mencoba membuka limit order {order_type}")
        ticker = binance.fetch_ticker(symbol)
        price = ticker["last"]

        # Tentukan harga limit untuk order
        if order_type == "BUY":
            limit_price = price * 0.999  # Harga limit 0.1% lebih rendah
            amount = trade_amount / limit_price
            order = binance.create_limit_buy_order(symbol, amount, limit_price)
        else:
            limit_price = price * 1.001  # Harga limit 0.1% lebih tinggi
            amount = trade_amount / price
            order = binance.create_limit_sell_order(symbol, amount, limit_price)

        send_telegram_message(f"🛒 *Limit {order_type} Order Dibuat*\n- Harga: {limit_price:.2f} USDT\n- Jumlah: {amount:.6f} BTC")
        logging.info(f"Limit order {order_type} ditempatkan pada harga {limit_price} USDT")

        # Tunggu order tereksekusi sebelum lanjut ke TP/SL
        return wait_for_order_execution(order['id'], order_type)
    except Exception as e:
        logging.error(f"Order {order_type} gagal: {e}")
        send_telegram_message(f"⚠️ *Order Gagal:* {e}")
        return None

# Fungsi Menunggu Order Tereksekusi
def wait_for_order_execution(order_id, order_type):
    try:
        while True:
            order = binance.fetch_order(order_id, symbol)
            if order['status'] == 'closed':
                entry_price = order['price']
                send_telegram_message(f"✅ *{order_type} Order Tereksekusi*\n- Harga: {entry_price} USDT")
                return entry_price
            time.sleep(5)  # Cek setiap 5 detik
    except Exception as e:
        logging.error(f"Error saat menunggu order tereksekusi: {e}")
        return None

# Fungsi Cek TP dan SL (Threading)
def check_tp_sl(entry_price):
    def monitor_price():
        while True:
            try:
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

                time.sleep(5)
            except Exception as e:
                logging.error(f"Error saat memantau TP/SL: {e}")
                break

    thread = threading.Thread(target=monitor_price)
    thread.daemon = True
    thread.start()

# Fungsi untuk menjalankan bot trading
def trading_bot():
    while True:
        try:
            spot_balance = check_balance()
            if spot_balance >= trade_amount:
                current_price = binance.fetch_ticker(symbol)["last"]
                logging.info(f"Harga saat ini: {current_price} USDT")

                entry_price = place_order("BUY")
                if entry_price:
                    check_tp_sl(entry_price)
            else:
                logging.info("Saldo tidak mencukupi untuk membuka posisi.")

            time.sleep(60)
        except Exception as e:
            logging.error(f"Error utama: {e}")
            send_telegram_message(f"⚠️ *Error:* {e}")
            time.sleep(10)

# Eksekusi bot
trading_bot()
