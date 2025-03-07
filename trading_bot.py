import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from binance.client import Client
from config import *
import requests

# Load model & scaler dengan perbaikan 'mse'
try:
    model = tf.keras.models.load_model("lstm_model.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
except TypeError:
    model = tf.keras.models.load_model("lstm_model.h5", compile=False)
    model.compile(loss="mse", optimizer="adam")  # Kompile ulang jika error

scaler = joblib.load("scaler.pkl")
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def fetch_latest_data():
    """Mengambil data harga terbaru dari Binance."""
    klines = client.get_klines(symbol=PAIR, interval=TIMEFRAME, limit=50)
    close_prices = [float(k[4]) for k in klines]
    close_prices = scaler.transform(np.array(close_prices).reshape(-1, 1))
    return np.array([close_prices])

def predict_price():
    """Memprediksi harga dengan model AI."""
    X_input = fetch_latest_data()
    prediction = model.predict(X_input)
    return scaler.inverse_transform(prediction)[0][0]

def send_telegram_message(message):
    """Mengirim pesan notifikasi ke Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

def trade():
    """Eksekusi trading berdasarkan prediksi AI."""
    predicted_price = predict_price()
    last_price = float(client.get_symbol_ticker(symbol=PAIR)["price"])

    TP = last_price * 1.015  # Take Profit 1.5%
    SL = last_price * 0.95   # Stop Loss 5%

    if predicted_price >= TP:  # **Kondisi BUY**
        order = client.order_market_buy(symbol=PAIR, quoteOrderQty=TRADE_AMOUNT_USDT)
        send_telegram_message(f"📈 BUY Order Executed at {last_price}")

    elif predicted_price <= SL:  # **Kondisi SELL**
        balance = client.get_asset_balance(asset="BTC")["free"]
        if float(balance) > 0:
            order = client.order_market_sell(symbol=PAIR, quantity=balance)  # **Sell semua BTC**
            send_telegram_message(f"📉 SELL Order Executed at {last_price}")

if __name__ == "__main__":
    trade()
