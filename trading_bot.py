import ccxt
import time
import numpy as np
import requests
import pandas as pd
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
min_balance = 10   # Minimal saldo USDT
trade_amount = 5   # Order 5 USDT per transaksi

# Fungsi Kirim Notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Error mengirim pesan Telegram: {e}")

# Mengambil data harga untuk LSTM
def get_training_data():
    ohlcv = binance.fetch_ohlcv(symbol, timeframe="15m", limit=500)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["close"] = df["close"].astype(float)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df["close"].values.reshape(-1, 1))

    sequence_length = 50
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Training Model LSTM
def train_lstm_model():
    X, y, scaler = get_training_data()

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    return model, scaler

# Prediksi Harga dengan LSTM
def predict_price(model, scaler):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe="15m", limit=50)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["close"] = df["close"].astype(float)

    last_50_closes = df["close"].values.reshape(-1, 1)
    scaled_data = scaler.transform(last_50_closes)

    X_input = np.array([scaled_data])
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

    predicted_price = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_price)

    return predicted_price[0][0]

# Jalankan bot
def trading_bot():
    print("🔄 Training LSTM Model...")
    model, scaler = train_lstm_model()

    while True:
        try:
            predicted_price = predict_price(model, scaler)
            current_price = binance.fetch_ticker(symbol)["last"]

            print(f"Predicted Price: {predicted_price:.2f}, Current Price: {current_price:.2f}")

            if predicted_price > current_price:
                print("🔹 AI Signal: BUY")
                send_telegram_message("🤖 *AI Signal: BUY* 🚀")
            elif predicted_price < current_price:
                print("🔻 AI Signal: SELL")
                send_telegram_message("🤖 *AI Signal: SELL* 📉")
            else:
                print("⏸️ AI Signal: HOLD")
                send_telegram_message("🤖 *AI Signal: HOLD* ⏳")

            time.sleep(60)
        except Exception as e:
            send_telegram_message(f"⚠️ *Error:* {e}")
            time.sleep(10)

# Eksekusi bot
trading_bot()
