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
tp_percentage =  5 / 100 # TP +5%
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
        return balance['total'].get('USDT', 0)
    except Exception as e:
        logging.error(f"Error mengecek saldo: {e}")
        send_telegram_message(f"⚠️ *Error mengecek saldo:* {e}")
        return 0

# Fungsi Membuka Market Order
def place_market_order(order_type):
    try:
        ticker = binance.fetch_ticker(symbol)
        price = ticker["last"]
        
        if order_type == "BUY":
            amount = trade_amount / price
            order = binance.create_market_buy_order(symbol, amount)
        else:
            amount = trade_amount / price
            order = binance.create_market_sell_order(symbol, amount)

        # Ambil harga eksekusi dari transaksi terakhir
        trades = binance.fetch_my_trades(symbol)
        entry_price = trades[-1]['price']

        send_telegram_message(f"📈 *{order_type} Order Executed*\n- Harga: {entry_price:.2f} USDT\n- TP: {entry_price * (1 + tp_percentage):.2f} USDT\n- SL: {entry_price * (1 - sl_percentage):.2f} USDT")
        logging.info(f"Market order {order_type} berhasil dieksekusi pada harga {entry_price} USDT")

        return entry_price
    except Exception as e:
        logging.error(f"Market order {order_type} gagal: {e}")
        send_telegram_message(f"⚠️ *Market Order Gagal:* {e}")
        return None

# Fungsi Memantau TP dan SL
def check_tp_sl(entry_price):
    def monitor_price():
        while True:
            try:
                ticker = binance.fetch_ticker(symbol)
                current_price = ticker['last']
                logging.info(f"Memeriksa harga: {current_price:.2f} USDT")

                if current_price >= entry_price * (1 + tp_percentage):
                    place_market_order("SELL")  # TP Terpenuhi
                    send_telegram_message(f"✅ *Take Profit Tercapai!* 🚀\n- Harga Jual: {current_price:.2f} USDT")
                    break
                elif current_price <= entry_price * (1 - sl_percentage):
                    place_market_order("SELL")  # SL Terpicu
                    send_telegram_message(f"⚠️ *Stop Loss Terpicu!* 📉\n- Harga Jual: {current_price:.2f} USDT")
                    break

                time.sleep(5)
            except Exception as e:
                logging.error(f"Error saat memantau TP/SL: {e}")
                send_telegram_message(f"⚠️ *Error TP/SL:* {e}")
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
                entry_price = place_market_order("BUY")
                if entry_price:
                    check_tp_sl(entry_price)
            else:
                logging.info("Saldo tidak cukup, menunggu saldo tersedia...")

            time.sleep(60)
        except Exception as e:
            logging.error(f"Error utama: {e}")
            send_telegram_message(f"⚠️ *Error utama:* {e}")
            time.sleep(10)

# Jalankan bot
trading_bot()
