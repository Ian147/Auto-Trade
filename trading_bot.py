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
logging.basicConfig(filename='trading_signal.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Konfigurasi API Binance
api_key = "API_KEY_ANDA"
api_secret = "API_SECRET_ANDA"

# Konfigurasi API Telegram
telegram_token = "TELEGRAM_BOT_TOKEN"
telegram_chat_id = "TELEGRAM_CHAT_ID"

# Inisialisasi Binance
binance = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"}
})

# Pair yang diperdagangkan
symbol = "BTC/USDT"

# Fungsi Kirim Notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error mengirim pesan Telegram: {e}")

# Fungsi Melatih Model LSTM
def train_lstm_model():
    try:
        historical_data = binance.fetch_ohlcv(symbol, timeframe='15m', limit=1000)
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

        best_mae = float('inf')

        for epoch in range(50):  # Batasi 50 epoch maksimal
            model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)

            if mae < best_mae:
                best_mae = mae

            if mae < 0.1:
                send_telegram_message(f"📊 *Model Akurasi*: {mae:.4f}")
                break

        return model, scaler
    except Exception as e:
        logging.error(f"Error saat melatih model LSTM: {e}")
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

# Fungsi menjalankan bot sinyal trading
def trading_signal_bot():
    model, scaler = train_lstm_model()

    while True:
        current_price = binance.fetch_ticker(symbol)["last"]
        predicted_price = predict_price(model, scaler)

        send_telegram_message(f"🔮 *Prediksi Harga BTC/USDT*:\n- Harga Saat Ini: {current_price:.2f} USDT\n- Prediksi Harga: {predicted_price:.2f} USDT")

        time.sleep(900)  # Tunggu 15 menit

# Eksekusi bot sinyal
trading_signal_bot()
