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

# Konfigurasi Logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    except Exception as e:
        logging.error(f"Error mengirim pesan Telegram: {e}")

# Fungsi Mengecek Saldo
def check_balance():
    try:
        balance = binance.fetch_balance()
        spot_balance = balance['total'].get('USDT', 0)
        logging.info(f"Saldo spot: {spot_balance} USDT")
        return spot_balance
    except Exception as e:
        logging.error(f"Error saat mengecek saldo: {e}")
        return 0

# Fungsi Open Order dengan Limit Order
def place_order(order_type):
    try:
        ticker = binance.fetch_ticker(symbol)
        price = ticker["last"]

        limit_price = price * 1.001 if order_type == "BUY" else price * 0.999
        amount = trade_amount / price  # Konversi dari USDT ke BTC

        if order_type == "BUY":
            order = binance.create_limit_buy_order(symbol, amount, limit_price)
        else:
            order = binance.create_limit_sell_order(symbol, amount, limit_price)

        send_telegram_message(f"📈 *{order_type} Order Placed*\n- Harga: {limit_price:.2f} USDT")
        return limit_price
    except Exception as e:
        logging.error(f"Order {order_type} gagal: {e}")
        return None

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

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        epoch = 0
        best_mae = float('inf')

        while epoch < 10:  # Batasi jumlah epoch untuk efisiensi
            epoch += 1
            model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)

            logging.info(f"Epoch {epoch} - MAE: {mae}")

            if mae < best_mae:
                best_mae = mae
            
            if mae < 0.1:
                logging.info(f"Model sudah cukup akurat! MAE: {mae}")
                break

        return model, scaler
    except Exception as e:
        logging.error(f"Error saat melatih model LSTM: {e}")
        return None, None

# Fungsi Prediksi Harga
def predict_price(model, scaler):
    try:
        latest_data = binance.fetch_ohlcv(symbol, timeframe='1m', limit=60)
        df = pd.DataFrame(latest_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        close_prices = df['close'].values.reshape(-1, 1)

        scaled_data = scaler.transform(close_prices)
        X_input = scaled_data[-60:].reshape(1, 60, 1)

        predicted_price = model.predict(X_input)
        predicted_price = scaler.inverse_transform(predicted_price)
        return predicted_price[0][0]
    except Exception as e:
        logging.error(f"Error saat memprediksi harga: {e}")
        return 0

# Fungsi Menjalankan Bot Trading
def trading_bot():
    model, scaler = train_lstm_model()
    if model is None or scaler is None:
        logging.error("Model gagal dilatih, bot berhenti.")
        return

    while True:
        try:
            spot_balance = check_balance()

            if spot_balance >= trade_amount:
                current_price = binance.fetch_ticker(symbol)["last"]
                predicted_price = predict_price(model, scaler)

                send_telegram_message(f"🔮 *Prediksi Harga*: {predicted_price:.2f} USDT")

                if predicted_price > current_price * 1.01:
                    place_order("BUY")

            time.sleep(60)  # Tunggu 1 menit sebelum cek ulang
        except Exception as e:
            logging.error(f"Error utama: {e}")
            time.sleep(10)

# Eksekusi bot
trading_bot()
