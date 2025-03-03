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
import xgboost as xgb
import threading

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
tp_percentage = 2 / 100  # TP +2%
sl_percentage = 1 / 100  # SL -1%

# Variabel Global
entry_price = None  # Harga beli terakhir
last_trade_time = 0  # Waktu terakhir melakukan transaksi

# Fungsi Kirim Notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error mengirim pesan Telegram: {e}")

# Fungsi Kirim Notifikasi *Hold* (30 Menit)
def send_hold_status():
    global last_trade_time

    while True:
        time.sleep(1800)  # 30 menit

        if entry_price is None:
            reason = "🔍 Bot masih menunggu sinyal yang kuat sebelum membuka order."
            send_telegram_message(f"⏳ *Status: HOLD* ⏳\n{reason}")

# Fungsi Kirim Notifikasi *Open Posisi* (10 Menit)
def send_trade_status():
    global entry_price

    while True:
        time.sleep(600)  # 10 menit

        if entry_price is not None:
            ticker = binance.fetch_ticker(symbol)
            current_price = ticker['last']
            tp_price = entry_price * (1 + tp_percentage)
            sl_price = entry_price * (1 - sl_percentage)

            send_telegram_message(f"""
📢 *Status: OPEN TRADE*
- 🎯 TP: {tp_price:.2f} USDT
- 🛑 SL: {sl_price:.2f} USDT
- 📊 Harga Sekarang: {current_price:.2f} USDT
⏳ Menunggu TP atau SL...
""")

# Fungsi Open Order
def place_order(order_type):
    global entry_price

    try:
        ticker = binance.fetch_ticker(symbol)
        price = ticker["last"]
        qty = trade_amount / price  

        if order_type == "BUY":
            order = binance.create_market_buy_order(symbol, qty)
        else:
            order = binance.create_market_sell_order(symbol, qty)

        entry_price = binance.fetch_my_trades(symbol)[-1]['price']
        send_telegram_message(f"📈 *{order_type} Order Executed*\n- Harga: {entry_price} USDT\n- TP: {entry_price * (1 + tp_percentage):.2f} USDT\n- SL: {entry_price * (1 - sl_percentage):.2f} USDT")
        return entry_price
    except Exception as e:
        logging.error(f"Order {order_type} gagal: {e}")
        return None

# Fungsi Cek TP dan SL
def check_tp_sl():
    global entry_price

    while True:
        if entry_price is None:
            time.sleep(10)
            continue

        try:
            ticker = binance.fetch_ticker(symbol)
            current_price = ticker['last']

            tp_price = entry_price * (1 + tp_percentage)
            sl_price = entry_price * (1 - sl_percentage)

            if current_price >= tp_price:
                place_order("SELL")
                send_telegram_message(f"✅ *Take Profit Tercapai!* 🚀\n- Harga Jual: {current_price:.2f} USDT")
                entry_price = None  # Reset setelah jual
            elif current_price <= sl_price:
                place_order("SELL")
                send_telegram_message(f"⚠️ *Stop Loss Terpicu!* 📉\n- Harga Jual: {current_price:.2f} USDT")
                entry_price = None  # Reset setelah jual

            time.sleep(10)
        except Exception as e:
            logging.error(f"Error saat memantau TP/SL: {e}")

# Fungsi Menjalankan Bot
def trading_bot():
    global entry_price

    # Jalankan notifikasi status di thread terpisah
    threading.Thread(target=send_hold_status, daemon=True).start()
    threading.Thread(target=send_trade_status, daemon=True).start()

    while True:
        if entry_price is None:
            ticker = binance.fetch_ticker(symbol)
            current_price = ticker['last']

            # Logika Beli
            if some_buy_condition:  # Ganti dengan logika beli yang tepat
                entry_price = place_order("BUY")
        
        time.sleep(900)  # Tunggu 15 menit

# Jalankan bot dalam thread terpisah
trading_thread = threading.Thread(target=trading_bot, daemon=True)
trading_thread.start()

# Jalankan TP/SL Monitoring di thread terpisah
tp_sl_thread = threading.Thread(target=check_tp_sl, daemon=True)
tp_sl_thread.start()
