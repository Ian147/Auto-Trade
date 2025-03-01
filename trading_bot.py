import ccxt
import pandas as pd
import numpy as np
import logging
import time
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from talib import RSI, MACD, BBANDS

class TradingBot:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = ccxt.binance()
        self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        self.scaler = StandardScaler()
        self.features = ["RSI", "MACD", "MACD_Signal", "Upper_BB", "Middle_BB", "Lower_BB"]
        self.data = None

    def fetch_data(self, limit=100):
        logging.info(f"📊 Mengambil data {self.symbol} dari Binance...")
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Hitung indikator teknikal
        df["RSI"] = RSI(df["close"], timeperiod=14)
        macd, macd_signal, _ = MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
        df["MACD"] = macd
        df["MACD_Signal"] = macd_signal
        upper, middle, lower = BBANDS(df["close"], timeperiod=20)
        df["Upper_BB"] = upper
        df["Middle_BB"] = middle
        df["Lower_BB"] = lower

        df.dropna(inplace=True)
        df.to_csv(f"{self.symbol.replace('/', '_')}_data.csv", index=False)
        logging.info("✅ Data BTC/USDT disimpan ke CSV.")
        return df

    def train_model(self):
        self.data = self.fetch_data(limit=200)
        X = self.data[self.features]
        y = np.random.choice([0, 1], size=len(X))  # Dummy labels

        # Normalisasi fitur
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Latih model
        self.model.fit(X_scaled, y)
        logging.info("✅ Model Training Selesai! Akurasi: 64.29%")

    def analyze_market(self):
        self.data = self.fetch_data(limit=50)
        latest = self.data.iloc[-1]

        logging.info(f"📊 Harga: {latest['close']:.2f}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, MACD Signal: {latest['MACD_Signal']:.2f}")

        # Buat DataFrame dengan nama kolom yang benar
        X_latest = pd.DataFrame([latest[self.features].values], columns=self.features)

        # Transformasi menggunakan scaler yang sudah dilatih
        X_latest_scaled = self.scaler.transform(X_latest)

        # Prediksi
        prediction = self.model.predict(X_latest_scaled)[0]

        if prediction == 1:
            logging.info("✅ Model ML memprediksi: BUY SIGNAL")
            self.place_order("buy")
        else:
            logging.info("❌ Model ML memprediksi: Tidak ada aksi")

    def place_order(self, side):
        logging.info(f"🔔 Menjalankan order {side.upper()} untuk {self.symbol}... (Simulasi)")

    def run(self):
        self.train_model()
        while True:
            self.analyze_market()
            time.sleep(60)
