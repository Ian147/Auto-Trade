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

# Fungsi Mengecek Saldo
def check_balance(asset):
    try:
        balance = binance.fetch_balance()
        return balance['total'].get(asset, 0)
    except Exception as e:
        logging.error(f"Error mengecek saldo {asset}: {e}")
        return 0

# Fungsi Menghitung RSI
def calculate_rsi(data, period=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fungsi Melatih Model LSTM
def train_lstm_model():
    try:
        historical_data = binance.fetch_ohlcv(symbol, timeframe='1h', limit=1000)
        df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        close_prices = df['close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=50, batch_size=32, verbose=1)
        return model, scaler
    except Exception as e:
        logging.error(f"Error saat melatih model LSTM: {e}")
        return None, None

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

# Fungsi Open Order
def place_order(order_type):
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

# Fungsi menjalankan bot
def trading_bot():
    lstm_model, lstm_scaler = train_lstm_model()

    while True:
        usdt_balance = check_balance("USDT")
        btc_balance = check_balance("BTC")
        lstm_predicted_price = predict_price_lstm(lstm_model, lstm_scaler)
        current_price = binance.fetch_ticker(symbol)["last"]

        # Hitung RSI
        historical_data = binance.fetch_ohlcv(symbol, timeframe='15m', limit=100)
        df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['RSI'] = calculate_rsi(df['close'])
        rsi = df['RSI'].iloc[-1]

        if lstm_predicted_price > current_price * 1.01 and rsi < 30 and usdt_balance >= trade_amount:
            entry_price = place_order("BUY")
            if entry_price:
                send_telegram_message(f"✅ *BUY Order Placed* at {entry_price:.2f} USDT")

        time.sleep(900)

# Eksekusi bot
trading_bot()
