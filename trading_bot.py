import logging
import ccxt
import time
import numpy as np
import requests
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from threading import Thread
import os

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

# Nama file model AI
model_filename = "lstm_trading_model.h5"

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
            qty = usdt_balance / price if usdt_balance >= trade_amount else trade_amount / price
            order = binance.create_market_buy_order(symbol, qty)

        elif order_type == "SELL":
            btc_balance = check_balance('BTC')
            if btc_balance > 0:
                order = binance.create_market_sell_order(symbol, btc_balance)
            else:
                return None

        entry_price = binance.fetch_my_trades(symbol)[-1]['price']
        send_telegram_message(f"📈 *{order_type} Order Executed*\n- Harga: {entry_price} USDT\n- TP: {entry_price * (1 + tp_percentage):.2f} USDT\n- SL: {entry_price * (1 - sl_percentage):.2f} USDT")
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
                    break
                elif current_price <= entry_price * (1 - sl_percentage):
                    place_order("SELL")
                    send_telegram_message(f"⚠️ *Stop Loss Terpicu!* 📉\n- Harga Jual: {current_price:.2f} USDT")
                    break

                time.sleep(10)  # Cek harga setiap 10 detik
            except Exception as e:
                logging.error(f"Error saat memantau TP/SL: {e}")
                break

    thread = Thread(target=monitor_price)
    thread.daemon = True
    thread.start()

# Fungsi Membuat Model AI
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fungsi Melatih Model AI
def train_lstm_model():
    try:
        historical_data = binance.fetch_ohlcv(symbol, timeframe='15m', limit=5000)
        df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        close_prices = df['close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

        def create_dataset(data, time_step=60):
            X, y = [], []
            for i in range(time_step, len(data)):
                X.append(data[i-time_step:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        X_train, y_train = create_dataset(train_data)
        X_test, y_test = create_dataset(test_data)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        if os.path.exists(model_filename):
            model = load_model(model_filename)
        else:
            model = create_lstm_model((X_train.shape[1], 1))

        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        model.save(model_filename)
        return model, scaler
    except Exception as e:
        logging.error(f"Error saat melatih model AI: {e}")
        return None, None

# Fungsi Prediksi Harga
def predict_price(model, scaler):
    try:
        latest_data = binance.fetch_ohlcv(symbol, timeframe='15m', limit=60)
        df = pd.DataFrame(latest_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        close_prices = df['close'].values.reshape(-1, 1)

        scaled_data = scaler.transform(close_prices)
        X_input = scaled_data[-60:].reshape(1, 60, 1)

        predicted_price = model.predict(X_input)
        return scaler.inverse_transform(predicted_price)[0][0]
    except Exception as e:
        logging.error(f"Error saat prediksi: {e}")
        return 0

# Fungsi Menjalankan Bot
def trading_bot():
    model, scaler = train_lstm_model()
    while True:
        usdt_balance = check_balance('USDT')
        predicted_price = predict_price(model, scaler)
        current_price = binance.fetch_ticker(symbol)["last"]

        if usdt_balance >= trade_amount and predicted_price > current_price * 1.01:
            entry_price = place_order("BUY")
            if entry_price:
                check_tp_sl(entry_price)

        time.sleep(900)  # Tunggu 15 menit

# Jalankan bot
trading_bot()
