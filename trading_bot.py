import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import requests
from binance.client import Client
from data_fetcher import get_binance_ohlcv
from config import BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, PAIR, TRADE_AMOUNT_USDT

# ✅ Load model & scaler
model = tf.keras.models.load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def fetch_latest_data():
    """ Mengambil data terbaru untuk prediksi """
    df = get_binance_ohlcv(limit=60)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df_scaled = scaler.transform(df)  # Transformasi data
    return np.array([df_scaled])

def predict_price():
    """ Prediksi harga dengan model LSTM """
    X_input = fetch_latest_data()
    prediction = model.predict(X_input)
    price_pred = scaler.inverse_transform([[0, 0, 0, prediction[0][0], 0]])[0][3]
    return price_pred

def send_telegram_message(message):
    """ Mengirim pesan ke Telegram """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

def trade():
    """ Proses trading berdasarkan prediksi harga """
    pred_price = predict_price()
    current_price = float(client.get_symbol_ticker(symbol=PAIR)["price"])

    message = f"📢 Prediksi Harga: {pred_price:.2f} USDT\n🔹 Harga Saat Ini: {current_price:.2f} USDT"
    send_telegram_message(message)

# ✅ Jalankan bot
trade()
