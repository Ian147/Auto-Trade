import pandas as pd
import numpy as np
import tensorflow as tf
from binance.client import Client
import joblib
import time
from data_fetcher import get_binance_ohlcv
from config import API_KEY, API_SECRET

# Load model dan scaler
MODEL_PATH = "lstm_model.h5"
SCALER_PATH = "scaler.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

client = Client(API_KEY, API_SECRET)

def predict_next_close():
    df = get_binance_ohlcv("BTCUSDT", "15m", 60)  
    if df is None:
        return None

    scaled_data = scaler.transform(df[["open", "high", "low", "close", "volume"]])
    X_input = np.array([scaled_data])
    pred_scaled = model.predict(X_input)[0][0]

    # Invers transform hasil prediksi
    pred_price = scaler.inverse_transform([[0, 0, 0, pred_scaled, 0]])[0][3]
    return pred_price

def execute_trade():
    balance = client.get_asset_balance(asset="USDT")
    usdt_balance = float(balance["free"])

    if usdt_balance < 10:
        print("⚠️ Saldo USDT kurang untuk trading")
        return

    predicted_price = predict_next_close()
    if predicted_price is None:
        print("⚠️ Gagal mendapatkan prediksi harga")
        return

    current_price = float(client.get_symbol_ticker(symbol="BTCUSDT")["price"])

    if predicted_price > current_price * 1.005:  # Naik 0.5%
        qty = round(10 / current_price, 6)
        order = client.order_market_buy(symbol="BTCUSDT", quantity=qty)
        print(f"✅ Order BUY berhasil: {order}")

    elif predicted_price < current_price * 0.995:  # Turun 0.5%
        btc_balance = float(client.get_asset_balance(asset="BTC")["free"])
        if btc_balance > 0:
            order = client.order_market_sell(symbol="BTCUSDT", quantity=btc_balance)
            print(f"✅ Order SELL berhasil: {order}")

if __name__ == "__main__":
    while True:
        execute_trade()
        time.sleep(900)  # 15 menit
