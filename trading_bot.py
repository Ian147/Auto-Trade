import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from binance.client import Client
from config import *
import requests
import logging

# Setup logging
logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load model & scaler dengan custom_objects untuk menghindari error 'mse'
try:
    model = tf.keras.models.load_model("lstm_model.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
except TypeError:
    model = tf.keras.models.load_model("lstm_model.h5", compile=False)
    model.compile(loss="mse", optimizer="adam")  # Kompile ulang model jika perlu

scaler = joblib.load("scaler.pkl")
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def fetch_latest_data():
    """Mengambil data harga terbaru dari Binance."""
    klines = client.get_klines(symbol=PAIR, interval=TIMEFRAME, limit=50)
    close_prices = [float(k[4]) for k in klines]
    close_prices = scaler.transform(np.array(close_prices).reshape(-1, 1))
    return np.array([close_prices])

def predict_price():
    """Memprediksi harga berdasarkan model AI."""
    X_input = fetch_latest_data()
    prediction = model.predict(X_input)
    
    # Pastikan hasil prediksi valid
    if np.isnan(prediction).any():
        logging.error("Predicted price is NaN! Skipping trade.")
        return None

    return scaler.inverse_transform(prediction)[0][0]

def send_telegram_message(message):
    """Mengirim pesan ke Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload)

def round_lot_size(quantity, step_size):
    """Menyesuaikan jumlah order agar sesuai dengan aturan LOT_SIZE Binance."""
    return round(quantity // step_size * step_size, len(str(step_size).split(".")[1]) if "." in str(step_size) else 0)

def trade():
    """Fungsi utama untuk trading berdasarkan prediksi AI."""
    predicted_price = predict_price()
    
    if predicted_price is None:
        return  # Skip trade jika prediksi tidak valid

    last_price = float(client.get_symbol_ticker(symbol=PAIR)["price"])

    TP = last_price * 1.015  # +1.5% dari harga saat ini
    SL = last_price * 0.95   # -5% dari harga saat ini

    logging.info(f"Last Price: {last_price}, Predicted Price: {predicted_price}, TP: {TP}, SL: {SL}")

    # **Kondisi BUY**
    if predicted_price >= TP:  
        try:
            order = client.order_market_buy(symbol=PAIR, quoteOrderQty=TRADE_AMOUNT_USDT)
            send_telegram_message(f"📈 BUY Order Executed at {last_price}")
            logging.info(f"BUY Order Executed at {last_price}")
        except Exception as e:
            logging.error(f"BUY Order Failed: {e}")

    # **Kondisi SELL**
    elif predicted_price <= SL:  
        try:
            balance = float(client.get_asset_balance(asset="BTC")["free"])

            if balance > 0:
                # Dapatkan aturan LOT_SIZE Binance untuk pembulatan jumlah BTC
                exchange_info = client.get_symbol_info(PAIR)
                step_size = float([f for f in exchange_info["filters"] if f["filterType"] == "LOT_SIZE"][0]["stepSize"])
                qty_to_sell = round_lot_size(balance, step_size)

                if qty_to_sell > 0:
                    client.order_market_sell(symbol=PAIR, quantity=qty_to_sell)
                    send_telegram_message(f"📉 SELL Order Executed at {last_price}")
                    logging.info(f"SELL Order Executed at {last_price}")
                else:
                    logging.warning("SELL Order failed: Lot size too small")
            else:
                logging.warning("No BTC balance to sell.")
        except Exception as e:
            logging.error(f"SELL Order Failed: {e}")

if __name__ == "__main__":
    trade()
