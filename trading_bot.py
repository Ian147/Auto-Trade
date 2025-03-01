import time
import requests
import pandas as pd
import numpy as np
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - INFO - %(message)s")

# Konfigurasi API Binance
API_KEY = "YOUR_BINANCE_API_KEY"
API_SECRET = "YOUR_BINANCE_SECRET_KEY"
client = Client(API_KEY, API_SECRET)

# Konfigurasi Telegram
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=payload)

class TradingBot:
    def __init__(self, symbol="BTCUSDT", timeframe="1m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.scaler = StandardScaler()
        self.last_signal = None

    def fetch_data(self, limit=50):
        klines = client.get_klines(symbol=self.symbol, interval=self.timeframe, limit=limit)
        df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
        df["close"] = df["close"].astype(float)
        df["rsi"] = RSIIndicator(df["close"]).rsi()
        macd = MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df = df.dropna()
        return df

    def train_model(self):
        df = self.fetch_data(limit=200)
        df["label"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
        X = df[["rsi", "macd", "macd_signal"]]
        y = df["label"]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        acc = self.model.score(X_scaled, y) * 100
        logging.info(f"✅ Model Training Selesai! Akurasi: {acc:.2f}%")

    def predict(self):
        df = self.fetch_data(limit=50)
        X_new = df[["rsi", "macd", "macd_signal"]].tail(1)
        X_scaled = self.scaler.transform(X_new)
        prediction = self.model.predict(X_scaled)[0]
        return prediction

    def execute_trade(self):
        prediction = self.predict()
        price = float(client.get_symbol_ticker(symbol=self.symbol)["price"])
        tp1 = price * 1.005
        tp2 = price * 1.01
        sl = price * 0.995

        if prediction == 1 and self.last_signal != "BUY":
            self.last_signal = "BUY"
            send_telegram_message(f"🔔 Open BUY di {price:.2f}\n🎯 TP1: {tp1:.2f}, TP2: {tp2:.2f}\n🛑 SL: {sl:.2f}")
            logging.info(f"🟢 Open BUY at {price:.2f}, TP1: {tp1:.2f}, TP2: {tp2:.2f}, SL: {sl:.2f}")
        elif prediction == 0 and self.last_signal != "SELL":
            self.last_signal = "SELL"
            send_telegram_message(f"🔔 Open SELL di {price:.2f}\n🎯 TP1: {tp1:.2f}, TP2: {tp2:.2f}\n🛑 SL: {sl:.2f}")
            logging.info(f"🔴 Open SELL at {price:.2f}, TP1: {tp1:.2f}, TP2: {tp2:.2f}, SL: {sl:.2f}")
        else:
            logging.info("❌ Tidak ada aksi")

    def run(self):
        self.train_model()
        while True:
            self.execute_trade()
            time.sleep(60)

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
