import logging
import ccxt
import time
import requests
import pandas as pd
import numpy as np
from threading import Thread
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error

# Konfigurasi Logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Konfigurasi API Binance
api_key = "API_KEY_ANDA"
api_secret = "API_SECRET_ANDA"

# Konfigurasi API Telegram
telegram_token = "TELEGRAM_TOKEN_ANDA"
telegram_chat_id = "CHAT_ID_ANDA"

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

# Fungsi Mengecek Saldo Spot USDT
def check_balance():
    try:
        balance = binance.fetch_balance()
        return balance['total'].get('USDT', 0)
    except Exception as e:
        logging.error(f"Error mengecek saldo: {e}")
        send_telegram_message(f"⚠️ *Error mengecek saldo:* {e}")
        return 0

# Fungsi Membuka Limit Order
def place_limit_order(order_type):
    try:
        ticker = binance.fetch_ticker(symbol)
        price = ticker["last"]

        if order_type == "BUY":
            limit_price = price * 0.995  # Harga beli sedikit lebih rendah (-0.5%)
            amount = trade_amount / limit_price
            order = binance.create_limit_buy_order(symbol, amount, limit_price)
        else:
            limit_price = price * 1.005  # Harga jual sedikit lebih tinggi (+0.5%)
            amount = trade_amount / price
            order = binance.create_limit_sell_order(symbol, amount, limit_price)

        logging.info(f"Order {order_type} dibuat di harga {limit_price:.2f} USDT")
        send_telegram_message(f"📌 *Order {order_type} Placed*\n- Harga: {limit_price:.2f} USDT")
        
        return order, limit_price

    except Exception as e:
        logging.error(f"Error saat melakukan order {order_type}: {e}")
        send_telegram_message(f"⚠️ *Order {order_type} Gagal:* {e}")
        return None, None

# Fungsi Membuka TP dan SL sebagai Limit Order
def place_tp_sl_orders(entry_price, amount):
    try:
        tp_price = entry_price * (1 + tp_percentage)
        sl_price = entry_price * (1 - sl_percentage)

        # Buat Limit Order untuk TP dan SL
        tp_order = binance.create_limit_sell_order(symbol, amount, tp_price)
        sl_order = binance.create_limit_sell_order(symbol, amount, sl_price)

        logging.info(f"TP order dibuat di {tp_price:.2f}, SL order dibuat di {sl_price:.2f}")
        send_telegram_message(f"✅ *TP/SL Order Placed*\n- TP: {tp_price:.2f} USDT\n- SL: {sl_price:.2f} USDT")
    
    except Exception as e:
        logging.error(f"Error membuat TP/SL orders: {e}")
        send_telegram_message(f"⚠️ *Gagal membuat TP/SL orders:* {e}")

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
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        return model, scaler
    except Exception as e:
        logging.error(f"Error saat melatih model: {e}")
        send_telegram_message(f"⚠️ *Error saat melatih model:* {e}")
        return None, None

# Fungsi untuk menjalankan bot trading
def trading_bot():
    model, scaler = train_lstm_model()

    while True:
        try:
            spot_balance = check_balance()
            if spot_balance >= trade_amount:
                order, entry_price = place_limit_order("BUY")
                if entry_price:
                    amount = trade_amount / entry_price
                    place_tp_sl_orders(entry_price, amount)
            else:
                logging.info("Saldo tidak cukup, menunggu saldo tersedia...")

            time.sleep(60)
        except Exception as e:
            logging.error(f"Error utama: {e}")
            send_telegram_message(f"⚠️ *Error utama:* {e}")
            time.sleep(10)

# Eksekusi bot
trading_bot()
