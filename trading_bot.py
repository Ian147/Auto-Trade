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
tp_percentage = 1.5 / 100  # TP +1.5%
sl_percentage = 1 / 100    # SL -1%

# Fungsi Kirim Notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        logging.error(f"Error mengirim pesan Telegram: {e}")

# Fungsi untuk Mengecek Saldo Spot
def check_balance():
    balance = binance.fetch_balance()
    spot_balance = balance['total']['USDT']
    logging.info(f"Saldo spot: {spot_balance} USDT")
    send_telegram_message(f"💰 *Saldo Spot* : {spot_balance} USDT")  # Kirim saldo spot ke Telegram
    return spot_balance

# Fungsi Open Order
def place_order(order_type):
    try:
        logging.info(f"Mencoba untuk membuka order {order_type}")
        if order_type == "BUY":
            order = binance.create_market_buy_order(symbol, trade_amount / binance.fetch_ticker(symbol)["last"])
        else:
            order = binance.create_market_sell_order(symbol, trade_amount / binance.fetch_ticker(symbol)["last"])
        
        # Ambil harga eksekusi order terakhir
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
            ticker = binance.fetch_ticker(symbol)
            current_price = ticker['last']

            logging.info(f"Memeriksa harga: {current_price} USDT")

            if current_price >= entry_price * (1 + tp_percentage):
                place_order("SELL")
                send_telegram_message(f"✅ *Take Profit Tercapai!* 🚀\n- Harga Jual: {current_price:.2f} USDT")
                break
            elif current_price <= entry_price * (1 - sl_percentage):
                place_order("SELL")
                send_telegram_message(f"⚠️ *Stop Loss Terpicu!* 📉\n- Harga Jual: {current_price:.2f} USDT")
                break

            time.sleep(5)  # Cek harga setiap 5 detik

    thread = threading.Thread(target=monitor_price)
    thread.start()

# Fungsi Melatih Model LSTM
def train_lstm_model():
    # Mengambil data historis dari Binance
    historical_data = binance.fetch_ohlcv(symbol, timeframe='1h', limit=1000)
    df = pd.DataFrame(historical_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Memilih hanya harga penutupan untuk pelatihan
    close_prices = df['close'].values.reshape(-1, 1)

    # Normalisasi harga
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Membagi data untuk pelatihan dan pengujian
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    # Membuat dataset untuk pelatihan
    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    # Reshape data untuk LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Membuat model LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Melatih model hingga akurasi 90%
    epoch = 0
    target_accuracy = 0.90  # Target akurasi 90%
    best_mae = float('inf')  # Variabel untuk menyimpan MAE terbaik

    while True:
        epoch += 1
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        
        # Prediksi harga menggunakan data uji
        y_pred = model.predict(X_test)

        # Menghitung MAE
        mae = mean_absolute_error(y_test, y_pred)
        logging.info(f"Epoch {epoch} - MAE: {mae}")

        # Cek jika MAE cukup rendah (akurasi cukup tinggi)
        if mae < best_mae:
            best_mae = mae
        
        # Jika MAE menunjukkan akurasi di atas 90%, berhenti melatih
        if mae < 0.1:  # Misalnya, MAE yang rendah menunjukkan akurasi tinggi
            logging.info(f"Model mencapai akurasi yang diinginkan dengan MAE: {mae}. Model siap untuk dijalankan!")
            send_telegram_message(f"📊 *Model Akurasi*: {mae:.4f}")  # Kirim akurasi model ke Telegram
            break

    return model, scaler

# Fungsi Prediksi Harga Menggunakan Model LSTM
def predict_price(model, scaler):
    latest_data = binance.fetch_ohlcv(symbol, timeframe='1m', limit=60)
    df = pd.DataFrame(latest_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    close_prices = df['close'].values.reshape(-1, 1)

    # Normalisasi data
    scaled_data = scaler.transform(close_prices)

    # Persiapkan data untuk prediksi
    X_input = scaled_data[-60:].reshape(1, 60, 1)

    # Prediksi harga
    predicted_price = model.predict(X_input)

    # Mengembalikan harga yang diprediksi dalam bentuk yang dapat dimengerti
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]

# Fungsi untuk menjalankan bot trading
def trading_bot():
    model, scaler = train_lstm_model()  # Melatih model sebelum memulai trading

    while True:
        try:
            # Mengecek saldo sebelum membuka posisi
            spot_balance = check_balance()

            if spot_balance >= trade_amount:
                # Cek harga saat ini
                current_price = binance.fetch_ticker(symbol)["last"]
                logging.info(f"Harga saat ini: {current_price} USDT")

                # Prediksi harga menggunakan LSTM
                predicted_price = predict_price(model, scaler)
                logging.info(f"Harga yang diprediksi: {predicted_price} USDT")

                # Evaluasi sinyal (misalnya, sinyal beli jika prediksi harga lebih tinggi)
                send_telegram_message(f"🔮 *Akurasi Sinyal* : {predicted_price:.2f} USDT")  # Kirim akurasi sinyal ke Telegram

                if predicted_price > current_price * 1.01:  # Harga diprediksi naik
                    entry_price = place_order("BUY")
                    if entry_price:
                        check_tp_sl(entry_price)
            else:
                logging.info("Saldo tidak mencukupi untuk membuka posisi. Menunggu saldo tersedia...")

            time.sleep(60)  # Cek saldo dan sinyal setiap 1 menit
        except Exception as e:
            logging.error(f"Error utama: {e}")
            send_telegram_message(f"⚠️ *Error:* {e}")
            time.sleep(10)

# Eksekusi bot
trading_bot()
