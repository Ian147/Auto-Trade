import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from binance.client import Client
from config import *
import requests

# Load model & scaler dengan custom_objects untuk menghindari error 'mse'
try:
    model = tf.keras.models.load_model("lstm_model.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
except TypeError:
    model = tf.keras.models.load_model("lstm_model.h5", compile=False)
    model.compile(loss="mse", optimizer="adam")  # Kompile ulang model jika perlu

scaler = joblib.load("scaler.pkl")

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def fetch_latest_data():
    klines = client.get_klines(symbol=PAIR, interval=TIMEFRAME, limit=50)
    close_prices = [float(k[4]) for k in klines]
    close_prices = scaler.transform(np.array(close_prices).reshape(-1, 1))
    return np.array([close_prices])

def predict_price():
    X_input = fetch_latest_data()
    prediction = model.predict(X_input)
    return scaler.inverse_transform(prediction)[0][0]

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

def trade():
    predicted_price = predict_price()
    last_price = float(client.get_symbol_ticker(symbol=PAIR)["price"])

    TP = last_price * (1 + TP_PERCENT / 100)
    SL = last_price * (1 - SL_PERCENT / 100)

    if predicted_price > TP:  # Kondisi BUY
        order = client.order_market_buy(symbol=PAIR, quoteOrderQty=TRADE_AMOUNT_USDT)
        send_telegram_message(f"📈 BUY Order Executed at {last_price}")

    elif predicted_price < SL:  # Kondisi SELL
        balance = client.get_asset_balance(asset="BTC")["free"]
        if float(balance) > 0:
            order = client.order_market_sell(symbol=PAIR, quantity=balance)
            send_telegram_message(f"📉 SELL Order Executed at {last_price}")

if __name__ == "__main__":
    trade()
