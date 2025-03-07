import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from binance.client import Client
from config import *
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, filename='bot.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Load model & scaler
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

def get_min_order_quantity(symbol):
    exchange_info = client.get_symbol_info(symbol)
    for filter in exchange_info['filters']:
        if filter['filterType'] == 'LOT_SIZE':
            return float(filter['minQty'])
    return None

def trade():
    predicted_price = predict_price()
    last_price = float(client.get_symbol_ticker(symbol=PAIR)["price"])

    TP = last_price * (1 + TP_PERCENT / 100)  # Calculate TP price
    SL = last_price * (1 - SL_PERCENT / 100)  # Calculate SL price

    logging.info(f"Last Price: {last_price}, Predicted Price: {predicted_price}, TP: {TP}, SL: {SL}")

    # If predicted price is greater than the target TP (BUY Signal)
    if predicted_price > last_price:
        # Market Buy 10 USDT worth of BTC
        order = client.order_market_buy(symbol=PAIR, quoteOrderQty=TRADE_AMOUNT_USDT)
        send_telegram_message(f"📈 BUY Order Executed at {last_price}\n🔼 TP: {TP}\n🔻 SL: {SL}")
        logging.info(f"BUY Order Executed at {last_price}, TP: {TP}, SL: {SL}")
        
        # Monitor for TP or SL to trigger SELL
        while True:
            current_price = float(client.get_symbol_ticker(symbol=PAIR)["price"])
            if current_price >= TP:  # TP reached
                balance = client.get_asset_balance(asset="BTC")["free"]
                balance = float(balance)
                if balance > 0:
                    # Sell all BTC at market price when TP is reached
                    order = client.order_market_sell(symbol=PAIR, quantity=balance)
                    send_telegram_message(f"🎯 TP Reached: SELL Order Executed at {current_price}\n📊 Profit: {TP - last_price}")
                    logging.info(f"SELL Order Executed at {current_price}, Profit: {current_price - last_price}")
                break
            elif current_price <= SL:  # SL reached
                balance = client.get_asset_balance(asset="BTC")["free"]
                balance = float(balance)
                if balance > 0:
                    # Sell all BTC at market price when SL is hit
                    order = client.order_market_sell(symbol=PAIR, quantity=balance)
                    send_telegram_message(f"❌ SL Reached: SELL Order Executed at {current_price}\n📊 Loss: {last_price - SL}")
                    logging.info(f"SELL Order Executed at {current_price}, Loss: {last_price - SL}")
                break

if __name__ == "__main__":
    trade()
