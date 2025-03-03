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
import xgboost as xgb
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
tp_percentage = 2 / 100  # TP +2%
sl_percentage = 1 / 100  # SL -1%

# Fungsi Kirim Notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error mengirim pesan Telegram: {e}")

# Fungsi untuk Mengecek Saldo Spot
def check_balance():
    try:
        balance = binance.fetch_balance()
        spot_balance = balance['total']['USDT']
        logging.info(f"Saldo spot: {spot_balance} USDT")
        return spot_balance
    except Exception as e:
        logging.error(f"Error saat mengecek saldo: {e}")
        return 0

# Fungsi Open Order
def place_order(order_type):
    try:
        ticker = binance.fetch_ticker(symbol)
        price = ticker["last"]
        qty = trade_amount / price  # Konversi USDT ke BTC

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

    thread = threading.Thread(target=monitor_price)
    thread.daemon = True
    thread.start()

# Fungsi Melatih Model LSTM
def train_lstm_model():
    try:
        historical_data = binance.fetch_ohlcv(symbol, timeframe='1h', limit=1000)
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

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
        return model, scaler
    except Exception as e:
        logging.error(f"Error saat melatih model LSTM: {e}")
        return None, None

# Fungsi Melatih Model XGBoost
def train_xgboost_model():
    try:
        historical_data = binance.fetch_ohlcv(symbol, timeframe='15m', limit=1000)
        df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        close_prices = df['close'].values.reshape(-1, 1)
        features = df[['open', 'high', 'low', 'volume']].values
        target = close_prices[1:]  # Shifted target for prediction

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(features[:-1], target)
        return model
    except Exception as e:
        logging.error(f"Error saat melatih model XGBoost: {e}")
        return None

# Fungsi Prediksi Harga LSTM
def predict_price_lstm(model, scaler):
    try:
        latest_data = binance.fetch_ohlcv(symbol, timeframe='1h', limit=60)
        df = pd.DataFrame(latest_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        close_prices = df['close'].values.reshape(-1, 1)

        scaled_data = scaler.transform(close_prices)
        X_input = scaled_data[-60:].reshape(1, 60, 1)

        predicted_price = model.predict(X_input)
        return scaler.inverse_transform(predicted_price)[0][0]
    except Exception as e:
        logging.error(f"Error saat prediksi harga LSTM: {e}")
        return 0

# Fungsi Prediksi Harga XGBoost
def predict_price_xgboost(model):
    try:
        latest_data = binance.fetch_ohlcv(symbol, timeframe='15m', limit=60)
        df = pd.DataFrame(latest_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        features = df[['open', 'high', 'low', 'volume']].values
        predicted_price = model.predict(features[-1].reshape(1, -1))
        return predicted_price[0]
    except Exception as e:
        logging.error(f"Error saat prediksi harga XGBoost: {e}")
        return 0

# Fungsi menjalankan bot
def trading_bot():
    lstm_model, lstm_scaler = train_lstm_model()
    xgboost_model = train_xgboost_model()

    while True:
        spot_balance = check_balance()
        if spot_balance >= trade_amount:
            lstm_predicted_price = predict_price_lstm(lstm_model, lstm_scaler)
            xgboost_predicted_price = predict_price_xgboost(xgboost_model)
            current_price = binance.fetch_ticker(symbol)["last"]

            # Gabungkan hasil prediksi LSTM dan XGBoost
            average_predicted_price = (lstm_predicted_price + xgboost_predicted_price) / 2

            if average_predicted_price > current_price * 1.01:  # Prediksi lebih tinggi 1%
                entry_price = place_order("BUY")
                if entry_price:
                    check_tp_sl(entry_price)

        time.sleep(900)  # Tunggu 15 menit

# Eksekusi bot
trading_bot()
