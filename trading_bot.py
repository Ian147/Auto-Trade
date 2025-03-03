import numpy as np
import pandas as pd
import logging
import time
import requests
from binance.client import Client
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import *
from data_fetcher import get_binance_ohlcv

logging.basicConfig(level=logging.INFO)

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
model = load_model("lstm_model.h5")

def predict_price():
    """ Memprediksi harga menggunakan model LSTM """
    df = get_binance_ohlcv(100)
    X, _ = prepare_data(df)
    pred = model.predict(X[-1].reshape(1, 50, 1))
    return scaler.inverse_transform(pred)[0][0]

def send_telegram_message(message):
    """ Mengirim notifikasi Telegram """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

def place_order(order_type):
    """ Menjalankan market order di Binance """
    if order_type == "BUY":
        qty = round(TRADE_AMOUNT_USDT / predict_price(), 6)
        order = client.order_market_buy(symbol=PAIR, quantity=qty)
        send_telegram_message(f"✅ BUY {qty} {PAIR} @ {predict_price()}")
        return order
    elif order_type == "SELL":
        balance = float(client.get_asset_balance(asset="BTC")["free"])
        order = client.order_market_sell(symbol=PAIR, quantity=round(balance, 6))
        send_telegram_message(f"✅ SELL {balance} {PAIR} @ {predict_price()}")
        return order

def trading_bot():
    """ Bot Trading AI dengan LSTM """
    while True:
        try:
            price_now = predict_price()
            df = get_binance_ohlcv(2)
            price_last = df['close'].iloc[-1]
            
            if price_now > price_last * (1 + TP_PERCENT / 100):
                logging.info("🚀 Take Profit Triggered")
                place_order("SELL")
            
            elif price_now < price_last * (1 - SL_PERCENT / 100):
                logging.info("⚠️ Stop Loss Triggered")
                place_order("SELL")
            
            elif price_now > price_last:
                logging.info("📈 Buy Signal Detected")
                place_order("BUY")

        except Exception as e:
            logging.error(f"❌ Error: {e}")
        
        time.sleep(60)  # Cek setiap 1 menit

if __name__ == "__main__":
    trading_bot()
