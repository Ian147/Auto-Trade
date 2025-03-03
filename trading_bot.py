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
from sklearn.metrics import mean_absolute_error
import threading

# Konfigurasi Logging
logging.basicConfig(filename='trading_bot.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Konfigurasi API Binance
api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"

# Konfigurasi API Telegram
telegram_token = "YOUR_TELEGRAM_BOT_TOKEN"
telegram_chat_id = "YOUR_TELEGRAM_CHAT_ID"

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
def check_balance(asset):
    try:
        balance = binance.fetch_balance()
        return balance['total'].get(asset, 0)
    except Exception as e:
        logging.error(f"Error saat mengecek saldo {asset}: {e}")
        return 0

# Fungsi Open Order
def place_order(order_type):
    try:
        ticker = binance.fetch_ticker(symbol)
        price = ticker["last"]

        if order_type == "BUY":
            usdt_balance = check_balance('USDT')
            qty = trade_amount / price if usdt_balance >= trade_amount else 0
            if qty > 0:
                order = binance.create_market_buy_order(symbol, qty)
                logging.info(f"BUY Order Executed: {qty:.6f} BTC at {price:.2f} USDT")
                send_telegram_message(f"📈 *BUY Order Executed*\n- Qty: {qty:.6f} BTC\n- Harga: {price:.2f} USDT")
            else:
                logging.warning("Saldo USDT tidak cukup untuk melakukan pembelian.")
                return None

        elif order_type == "SELL":
            btc_balance = check_balance('BTC')
            if btc_balance > 0:
                order = binance.create_market_sell_order(symbol, btc_balance)
                logging.info(f"SELL Order Executed: {btc_balance:.6f} BTC at {price:.2f} USDT")
                send_telegram_message(f"📉 *SELL Order Executed*\n- Qty: {btc_balance:.6f} BTC\n- Harga: {price:.2f} USDT")
            else:
                logging.warning("Saldo BTC tidak cukup untuk melakukan penjualan.")
                return None

        entry_price = binance.fetch_my_trades(symbol)[-1]['price']
        return entry_price

    except Exception as e:
        logging.error(f"Order {order_type} gagal: {e}")
        return None

# Fungsi Cek TP dan SL
def check_tp_sl(entry_price):
    def monitor_price():
        while True:
            try:
                ticker = binance.fetch_ticker(symbol)
                current_price = ticker['last']

                if current_price >= entry_price * (1 + tp_percentage):
                    place_order("SELL")
                    send_telegram_message(f"✅ *Take Profit Tercapai!* 🚀\n- Harga Jual: {current_price:.2f} USDT")
                    logging.info(f"Take Profit tercapai: Jual di {current_price:.2f} USDT")
                    break
                elif current_price <= entry_price * (1 - sl_percentage):
                    place_order("SELL")
                    send_telegram_message(f"⚠️ *Stop Loss Terpicu!* 📉\n- Harga Jual: {current_price:.2f} USDT")
                    logging.info(f"Stop Loss terpicu: Jual di {current_price:.2f} USDT")
                    break

                time.sleep(10)  # Cek harga setiap 10 detik
            except Exception as e:
                logging.error(f"Error saat memantau TP/SL: {e}")
                break

    thread = threading.Thread(target=monitor_price)
    thread.daemon = True
    thread.start()

# Fungsi Prediksi Harga dengan AI
def predict_price():
    try:
        latest_data = binance.fetch_ohlcv(symbol, timeframe='15m', limit=5000)
        df = pd.DataFrame(latest_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        close_prices = df['close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        X = []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
        X = np.array(X).reshape(-1, 60, 1)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, scaled_data[60:], epochs=10, batch_size=32, verbose=0)

        predicted_price = model.predict(X[-1].reshape(1, 60, 1))
        return scaler.inverse_transform(predicted_price)[0][0]
    except Exception as e:
        logging.error(f"Error saat prediksi: {e}")
        return 0

# Fungsi menjalankan bot
def trading_bot():
    while True:
        usdt_balance = check_balance('USDT')
        btc_balance = check_balance('BTC')
        predicted_price = predict_price()
        current_price = binance.fetch_ticker(symbol)["last"]

        if usdt_balance >= trade_amount and predicted_price > current_price * 1.01:
            entry_price = place_order("BUY")
            if entry_price:
                check_tp_sl(entry_price)

        elif btc_balance > 0 and current_price >= entry_price * (1 + tp_percentage):
            place_order("SELL")

        time.sleep(900)  # Tunggu 15 menit

# Eksekusi bot
trading_bot()
