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
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
sl_percentage = 2 / 100    # SL -2%

# Fungsi Kirim Notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error mengirim pesan Telegram: {e}")
        # Retry send message (recurse) after some time
        time.sleep(5)
        send_telegram_message(message)

# Fungsi untuk Mengecek Saldo Spot
def check_balance():
    try:
        balance = binance.fetch_balance()
        spot_balance = balance['total']['USDT']
        logging.info(f"Saldo spot: {spot_balance} USDT")
        return spot_balance
    except Exception as e:
        logging.error(f"Error saat mengecek saldo: {e}")
        send_telegram_message(f"⚠️ *Error saat mengecek saldo:* {e}")
        return 0

# Fungsi Open Order
def place_order(order_type):
    try:
        logging.info(f"Mencoba untuk membuka order {order_type}")
        ticker = binance.fetch_ticker(symbol)
        price = ticker["last"]
        if order_type == "BUY":
            order = binance.create_market_buy_order(symbol, trade_amount / price)
        else:
            order = binance.create_market_sell_order(symbol, trade_amount / price)

        entry_price = binance.fetch_my_trades(symbol)[-1]['price']
        send_telegram_message(f"📈 *{order_type} Order Executed*\n- Harga: {entry_price} USDT\n- TP: {entry_price * (1 + tp_percentage):.2f} USDT\n- SL: {entry_price * (1 - sl_percentage):.2f} USDT")
        logging.info(f"Order {order_type} berhasil dieksekusi pada harga {entry_price} USDT")
        return entry_price
    except Exception as e:
        logging.error(f"Order {order_type} gagal: {e}")
        send_telegram_message(f"⚠️ *Order Gagal:* {e}")
        return None

# Fungsi Cek TP dan SL (Menggunakan Threading)
def check_tp_sl(entry_price):
    def monitor_price():
        while True:
            try:
                ticker = binance.fetch_ticker(symbol)
                current_price = ticker['last']
                logging.info(f"Memeriksa harga: {current_price} USDT")

                # Mengecek jika harga sudah mencapai TP atau SL
                if current_price >= entry_price * (1 + tp_percentage):
                    place_order("SELL")  # Menutup posisi untuk TP
                    send_telegram_message(f"✅ *Take Profit Tercapai!* 🚀\n- Harga Jual: {current_price:.2f} USDT")
                    break
                elif current_price <= entry_price * (1 - sl_percentage):
                    place_order("SELL")  # Menutup posisi untuk SL
                    send_telegram_message(f"⚠️ *Stop Loss Terpicu!* 📉\n- Harga Jual: {current_price:.2f} USDT")
                    break

                time.sleep(3600)  # Cek harga setiap 1 jam (3600 detik)
            except Exception as e:
                logging.error(f"Error saat memantau TP/SL: {e}")
                send_telegram_message(f"⚠️ *Error saat memantau TP/SL:* {e}")
                break

    thread = threading.Thread(target=monitor_price)
    thread.daemon = True  # Membuat thread berjalan di background
    thread.start()

# Fungsi Melatih Model LSTM
def train_lstm_model():
    try:
        historical_data = binance.fetch_ohlcv(symbol, timeframe='1h', limit=1000)  # Menggunakan interval 1 jam
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
        target_accuracy = 0.90
        best_mae = float('inf')

        while True:
            epoch += 1
            model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            logging.info(f"Epoch {epoch} - MAE: {mae}")

            if mae < best_mae:
                best_mae = mae
            
            if mae < 0.1:
                logging.info(f"Model mencapai akurasi yang diinginkan dengan MAE: {mae}. Model siap untuk dijalankan!")
                send_telegram_message(f"📊 *Model Akurasi*: {mae:.4f}")
                break

        return model, scaler
    except Exception as e:
        logging.error(f"Error saat melatih model LSTM: {e}")
        send_telegram_message(f"⚠️ *Error saat melatih model LSTM:* {e}")
        return None, None

# Fungsi Prediksi Harga Menggunakan Model LSTM
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
        send_telegram_message(f"⚠️ *Error saat memprediksi harga:* {e}")
        return 0

# Fungsi untuk menjalankan bot trading
def trading_bot():
    model, scaler = train_lstm_model()

    while True:
        try:
            spot_balance = check_balance()

            if spot_balance >= trade_amount:
                current_price = binance.fetch_ticker(symbol)["last"]
                logging.info(f"Harga saat ini: {current_price} USDT")

                predicted_price = predict_price(model, scaler)
                logging.info(f"Harga yang diprediksi: {predicted_price} USDT")

                send_telegram_message(f"🔮 *Akurasi Sinyal* : {predicted_price:.2f} USDT")

                if predicted_price > current_price * 1.01:
                    entry_price = place_order("BUY")
                    if entry_price:
                        check_tp_sl(entry_price)
            else:
                logging.info("Saldo tidak mencukupi untuk membuka posisi. Menunggu saldo tersedia...")

            time.sleep(3600)  # Interval setiap 1 jam
        except Exception as e:
            logging.error(f"Error utama: {e}")
            send_telegram_message(f"⚠️ *Error:* {e}")
            time.sleep(10)

# Eksekusi bot
trading_bot()
