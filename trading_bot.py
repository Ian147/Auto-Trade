import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from binance.client import Client
from config import *
import requests

# Load model & scaler
model = tf.keras.models.load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# Variabel tracking order
current_order = None
buy_price = None

def fetch_latest_data():
    """ Mengambil data harga terbaru dari Binance """
    klines = client.get_klines(symbol=PAIR, interval=TIMEFRAME, limit=50)
    close_prices = np.array([float(k[4]) for k in klines]).reshape(-1, 1)
    
    # Pastikan scaler bekerja dengan bentuk data yang benar
    close_prices_scaled = scaler.transform(close_prices)
    return np.expand_dims(close_prices_scaled, axis=0)  # (1, 50, 1)

def predict_price():
    """ Memprediksi harga menggunakan model LSTM """
    X_input = fetch_latest_data()
    prediction = model.predict(X_input)
    
    # Kembalikan skala prediksi ke harga asli
    return scaler.inverse_transform(prediction)[0][0]

def send_telegram_message(message):
    """ Mengirim pesan ke Telegram """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

def trade():
    global current_order, buy_price
    
    predicted_price = predict_price()
    last_price = float(client.get_symbol_ticker(symbol=PAIR)["price"])

    if current_order is None and predicted_price > last_price * 1.015:  # BUY jika prediksi naik > 1.5%
        order = client.order_market_buy(symbol=PAIR, quoteOrderQty=TRADE_AMOUNT_USDT)
        buy_price = float(order["fills"][0]["price"])  # Simpan harga beli
        send_telegram_message(f"📈 BUY Order Executed at {buy_price}")
        current_order = "BUY"

    elif current_order == "BUY":
        if last_price >= buy_price * 1.015:  # Take Profit (TP 1.5%)
            balance = float(client.get_asset_balance(asset="BTC")["free"])
            if balance > 0:
                order = client.order_market_sell(symbol=PAIR, quantity=balance)
                send_telegram_message(f"✅ TP HIT! SELL at {last_price}")
                current_order = None  # Reset status order
        
        elif last_price <= buy_price * 0.95:  # Stop Loss (SL 5%)
            balance = float(client.get_asset_balance(asset="BTC")["free"])
            if balance > 0:
                order = client.order_market_sell(symbol=PAIR, quantity=balance)
                send_telegram_message(f"❌ SL HIT! SELL at {last_price}")
                current_order = None  # Reset status order

if __name__ == "__main__":
    trade()
