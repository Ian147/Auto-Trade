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
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)

# Inisialisasi Binance Client
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# Muat model LSTM
try:
    model = load_model("lstm_model.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    logging.info("✅ Model LSTM berhasil dimuat!")
except Exception as e:
    logging.error(f"❌ Gagal memuat model: {e}")
    exit()

# Inisialisasi MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

def prepare_data(df, lookback=50):
    """ Menyiapkan data untuk prediksi dengan LSTM """
    data = df['close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data)

    X = []
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:i+lookback])

    return np.array(X)

def predict_price():
    """ Memprediksi harga menggunakan model LSTM """
    df = get_binance_ohlcv(100)
    if df is None or df.empty:
        logging.warning("⚠️ Data tidak cukup untuk prediksi!")
        return None

    X = prepare_data(df)
    if len(X) == 0:
        logging.warning("⚠️ Data tidak cukup untuk prediksi!")
        return None

    pred = model.predict(X[-1].reshape(1, 50, 1))
    predicted_price = scaler.inverse_transform(pred)[0][0]
    return predicted_price

def send_telegram_message(message):
    """ Mengirim notifikasi Telegram """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        logging.error(f"❌ Gagal mengirim Telegram: {e}")

def get_balance(asset):
    """ Mendapatkan saldo aset tertentu """
    try:
        balance = client.get_asset_balance(asset=asset)
        return float(balance["free"]) if balance else 0.0
    except Exception as e:
        logging.error(f"❌ Gagal mendapatkan saldo {asset}: {e}")
        return 0.0

def round_step_size(quantity, step_size=0.00001):
    """ Membulatkan quantity sesuai aturan stepSize Binance """
    return round(quantity - (quantity % step_size), 8)

def place_order(order_type):
    """ Menjalankan market order di Binance """
    try:
        if order_type == "BUY":
            usdt_balance = get_balance("USDT")
            if usdt_balance < TRADE_AMOUNT_USDT:
                logging.warning("⚠️ Saldo USDT tidak cukup untuk BUY.")
                return None

            price_now = predict_price()
            if price_now is None:
                logging.error("❌ Tidak bisa mengeksekusi order, prediksi harga tidak tersedia.")
                return None

            # Perhitungan jumlah pembelian
            qty = TRADE_AMOUNT_USDT / price_now  
            qty = round_step_size(qty)  # Sesuaikan dengan stepSize Binance

            order = client.order_market_buy(symbol=PAIR, quantity=qty)

            tp_price = round(price_now * (1.5 + TP_PERCENT / 100), 2)
            sl_price = round(price_now * (5 - SL_PERCENT / 100), 2)

            send_telegram_message(f"✅ BUY {qty} BTC @ {price_now}\n🎯 TP: {tp_price}\n🛑 SL: {sl_price}")
            return order

        elif order_type == "SELL":
            # Menjual seluruh saldo BTC
            btc_balance = get_balance("BTC")
            if btc_balance <= 0:
                logging.warning("⚠️ Tidak ada saldo BTC untuk dijual.")
                return None

            qty = round_step_size(btc_balance)  # Jual seluruh saldo BTC
            order = client.order_market_sell(symbol=PAIR, quantity=qty)

            send_telegram_message(f"✅ SELL {qty} BTC @ {predict_price()}")
            return order

    except Exception as e:
        logging.error(f"❌ Error dalam order {order_type}: {e}")
        return None

def trading_bot():
    """ Bot Trading AI dengan LSTM """
    while True:
        try:
            price_now = predict_price()
            if price_now is None:
                logging.warning("⚠️ Tidak bisa melanjutkan trading, harga tidak tersedia.")
                time.sleep(60)
                continue

            df = get_binance_ohlcv(2)
            if df is None or df.empty:
                logging.warning("⚠️ Data tidak tersedia, menunggu...")
                time.sleep(60)
                continue

            price_last = df['close'].iloc[-1]

            # Cek apakah harga sekarang lebih besar dari TP
            if price_now > price_last * (1.5 + TP_PERCENT / 100):
                logging.info("🚀 Take Profit Triggered")
                place_order("SELL")  # **SELL seluruh saldo BTC saat TP tercapai**

            # Cek apakah harga sekarang lebih kecil dari SL
            elif price_now < price_last * (5 - SL_PERCENT / 100):
                logging.info("⚠️ Stop Loss Triggered")
                place_order("SELL")  # **SELL seluruh saldo BTC saat SL tercapai**

            # Kondisi untuk membuka BUY
            elif price_now > price_last:
                logging.info("📈 Buy Signal Detected")
                place_order("BUY")

        except Exception as e:
            logging.error(f"❌ Error: {e}")

        time.sleep(60)  # Cek setiap 1 menit

if __name__ == "__main__":
    trading_bot()
