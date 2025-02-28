import time
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from utils import get_ohlcv, calculate_indicators, save_data
from config import SYMBOL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TradingBot:
    def __init__(self, symbol=SYMBOL):
        self.symbol = symbol
        self.model = None
        self.scaler = None
    
    # Latih model Machine Learning
    def train_model(self):
        df = get_ohlcv()
        df = calculate_indicators(df)
        df.dropna(inplace=True)

        # Label target: 1 (Beli) jika harga naik di candle berikutnya, 0 jika tidak
        df["Target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
        
        features = ["open", "high", "low", "close", "volume", "RSI", "MACD", "MACD_Signal", "Upper_BB", "Middle_BB", "Lower_BB"]
        X = df[features]
        y = df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = xgb.XGBClassifier()
        self.model.fit(X_train, y_train)

        accuracy = self.model.score(X_test, y_test)
        logging.info(f"✅ Model Training Selesai! Akurasi: {accuracy * 100:.2f}%")

    # Analisis sinyal trading menggunakan Machine Learning
    def analyze_market(self):
        df = get_ohlcv()
        df = calculate_indicators(df)
        df.dropna(inplace=True)

        latest = df.iloc[-1]
        X_latest = self.scaler.transform([latest[["open", "high", "low", "close", "volume", "RSI", "MACD", "MACD_Signal", "Upper_BB", "Middle_BB", "Lower_BB"]]])

        prediction = self.model.predict(X_latest)[0]
        logging.info(f"📊 Harga: {latest['close']}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, MACD Signal: {latest['MACD_Signal']:.2f}")

        if prediction == 1:
            logging.info("✅ Model ML memprediksi: BELI!")
            return "BUY"
        else:
            logging.info("❌ Model ML memprediksi: Tidak ada aksi")
            return "HOLD"

    # Jalankan strategi trading
    def run(self):
        logging.info("🚀 Memulai bot trading...")
        self.train_model()
        
        while True:
            action = self.analyze_market()
            if action == "BUY":
                logging.info("🛒 Eksekusi Order BELI (dummy)...")
                # Tambahkan fungsi order ke Binance jika ingin benar-benar trading
            time.sleep(3600)  # Jalan setiap 1 jam
