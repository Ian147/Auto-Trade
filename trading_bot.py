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

# Fungsi Kirim Notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error mengirim pesan Telegram: {e}")

# Fungsi Fetch Data dan Menambahkan Indikator Teknikal
def fetch_data():
    try:
        historical_data = binance.fetch_ohlcv(symbol, timeframe='1h', limit=1000)
        df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Indikator RSI
        df['price_change'] = df['close'].diff()
        gain = df['price_change'].where(df['price_change'] > 0, 0)
        loss = -df['price_change'].where(df['price_change'] < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Indikator MACD
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']

        # Bollinger Bands
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['STD_20'] = df['close'].rolling(window=20).std()
        df['Upper_Band'] = df['SMA_20'] + (df['STD_20'] * 2)
        df['Lower_Band'] = df['SMA_20'] - (df['STD_20'] * 2)

        return df.dropna()
    except Exception as e:
        logging.error(f"Error saat fetch data: {e}")
        send_telegram_message(f"⚠️ *Error Fetch Data:* {e}")
        return None

# Fungsi Melatih Model LSTM
def train_lstm_model():
    df = fetch_data()
    if df is None:
        return None, None

    feature_columns = ['close', 'RSI', 'MACD', 'Upper_Band', 'Lower_Band']
    data = df[feature_columns].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(feature_columns)))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(feature_columns)))

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], len(feature_columns))),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    epoch = 0
    best_mae = float('inf')

    while True:
        epoch += 1
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        if mae < best_mae:
            best_mae = mae
        
        if mae < 0.1:
            send_telegram_message(f"📊 *Model Akurasi*: {mae:.4f}")
            break

    return model, scaler

# Fungsi Prediksi Harga
def predict_price(model, scaler):
    df = fetch_data()
    if df is None:
        return None

    latest_data = df[['close', 'RSI', 'MACD', 'Upper_Band', 'Lower_Band']].values[-60:]
    scaled_data = scaler.transform(latest_data)
    X_input = np.array([scaled_data])

    predicted_price = model.predict(X_input)
    predicted_price = scaler.inverse_transform(np.hstack((predicted_price, np.zeros((predicted_price.shape[0], 4)))))[:, 0]
    return predicted_price[0]

# Fungsi Trading Bot
def trading_bot():
    model, scaler = train_lstm_model()
    if model is None or scaler is None:
        return

    while True:
        try:
            current_price = binance.fetch_ticker(symbol)["last"]
            predicted_price = predict_price(model, scaler)

            if predicted_price is None:
                continue

            send_telegram_message(f"🔮 *Prediksi*: {predicted_price:.2f} USDT | Harga Sekarang: {current_price:.2f} USDT")

            if predicted_price > current_price * 1.01:
                logging.info("Sinyal beli terkonfirmasi")
                binance.create_market_buy_order(symbol, trade_amount / current_price)
                send_telegram_message("📈 *BUY Order Executed!* 🚀")
            elif predicted_price < current_price * 0.99:
                logging.info("Sinyal jual terkonfirmasi")
                binance.create_market_sell_order(symbol, trade_amount / current_price)
                send_telegram_message("📉 *SELL Order Executed!* 📉")

            time.sleep(60)
        except Exception as e:
            logging.error(f"Error utama: {e}")
            send_telegram_message(f"⚠️ *Error:* {e}")
            time.sleep(10)

# Jalankan Bot
trading_bot()
