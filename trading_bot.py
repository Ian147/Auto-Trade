import logging
import pandas as pd
import numpy as np
import time
from binance.client import Client
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TradingBot:
    def __init__(self, symbol="BTCUSDT", timeframe="1h"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.client = Client(api_key="API_KEY", api_secret="API_SECRET")
        self.scaler = StandardScaler()
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric="logloss")
        self.take_profit_1 = 1.02  # 2% TP1
        self.take_profit_2 = 1.05  # 5% TP2
        self.stop_loss = 0.98      # 2% SL

    def fetch_data(self, limit=50):
        """Mengambil data candle dari Binance"""
        try:
            klines = self.client.get_klines(symbol=self.symbol, interval=self.timeframe, limit=limit)
            df = pd.DataFrame(klines, columns=["time", "open", "high", "low", "close", "volume",
                                               "close_time", "quote_asset_volume", "trades",
                                               "taker_base_vol", "taker_quote_vol", "ignore"])
            df["close"] = df["close"].astype(float)
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            return df
        except Exception as e:
            logging.error(f"❌ Error fetch_data: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df):
        """Menghitung indikator RSI dan MACD"""
        df["returns"] = df["close"].pct_change()
        df["rsi"] = 100 - (100 / (1 + df["returns"].rolling(14).mean() / df["returns"].rolling(14).std()))
        df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df.dropna(inplace=True)
        return df

    def train_model(self):
        """Melatih model Machine Learning"""
        df = self.fetch_data(limit=100)
        df = self.calculate_indicators(df)

        if df.empty:
            logging.error("❌ Data tidak tersedia untuk training.")
            return

        features = df[["rsi", "macd", "macd_signal"]]
        labels = np.where(df["returns"].shift(-1) > 0, 1, 0)

        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)

        self.model.fit(features_scaled, labels)
        acc = self.model.score(features_scaled, labels) * 100
        logging.info(f"✅ Model Training Selesai! Akurasi: {acc:.2f}%")

    def predict_signal(self):
        """Memprediksi sinyal trading"""
        df = self.fetch_data(limit=20)
        df = self.calculate_indicators(df)

        if df.empty:
            return None

        latest = df.iloc[-1]
        features = np.array([[latest["rsi"], latest["macd"], latest["macd_signal"]]])
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]

        logging.info(f"📊 Harga: {latest['close']:.2f}, RSI: {latest['rsi']:.2f}, MACD: {latest['macd']:.2f}")
        return "BUY" if prediction == 1 else "NO ACTION"

    def execute_trade(self, signal):
        """Eksekusi perdagangan berdasarkan sinyal"""
        if signal == "BUY":
            price = float(self.client.get_symbol_ticker(symbol=self.symbol)["price"])
            tp1 = price * self.take_profit_1
            tp2 = price * self.take_profit_2
            sl = price * self.stop_loss

            logging.info(f"✅ BUY order ditempatkan pada {price:.2f}, TP1: {tp1:.2f}, TP2: {tp2:.2f}, SL: {sl:.2f}")

    def run(self):
        """Menjalankan bot trading"""
        logging.info("🚀 Memulai bot trading...")
        self.train_model()

        while True:
            signal = self.predict_signal()
            if signal:
                self.execute_trade(signal)

            time.sleep(60)
