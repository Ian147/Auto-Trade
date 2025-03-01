import logging
import pandas as pd
import time
import talib
import xgboost as xgb
from binance.client import Client
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# API Key Binance (Ganti dengan milikmu)
API_KEY = "6ipqniXiFRmjwGsB8H9vUpgVTexAnsLZ2Ybi0DrLxSKKINMr42wCC8ex7rIrqNlj"
API_SECRET = "HeINMThVDiJuCaoZFvC16FNj0ZCx9uGs2BxkkS1qTB3PkGTmibXfba3l8DajJ3x0"

class TradingBot:
    def __init__(self, symbol, timeframe):
        self.client = Client(API_KEY, API_SECRET)
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.last_entry_price = None

    def fetch_data(self, limit=500):
        """Mengambil data historis dari Binance"""
        klines = self.client.get_klines(symbol=self.symbol, interval=self.timeframe, limit=limit)
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["open"] = df["open"].astype(float)
        return df

    def add_indicators(self, df):
        """Menambahkan indikator teknikal"""
        df["RSI"] = talib.RSI(df["close"], timeperiod=14)
        df["MACD"], df["MACD_signal"], _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
        df["EMA9"] = talib.EMA(df["close"], timeperiod=9)
        df["EMA21"] = talib.EMA(df["close"], timeperiod=21)
        df["ATR"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
        df["Stoch_K"], df["Stoch_D"] = talib.STOCH(df["high"], df["low"], df["close"])
        df.dropna(inplace=True)
        return df

    def train_model(self):
        """Melatih model Machine Learning"""
        logging.info("📈 Melatih model Machine Learning...")

        self.data = self.fetch_data()
        self.data = self.add_indicators(self.data)

        features = ["RSI", "MACD", "MACD_signal", "EMA9", "EMA21", "ATR", "Stoch_K", "Stoch_D"]
        X = self.data[features]
        y = (self.data["close"].shift(-1) > self.data["close"]).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, use_label_encoder=False)
        self.model.fit(X_train_scaled, y_train)

        predictions = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        logging.info(f"✅ Model Training Selesai! Akurasi: {accuracy * 100:.2f}%")

    def execute_trade(self, signal, price):
        """Eksekusi trade dengan Take Profit dan Stop Loss"""
        if signal == "BUY":
            self.last_entry_price = price
            take_profit_1 = price * 1.015
            take_profit_2 = price * 1.03
            stop_loss = price * 0.985

            logging.info(f"🔵 BUY order at {price:.2f}")
            logging.info(f"🎯 Take Profit 1: {take_profit_1:.2f}, Take Profit 2: {take_profit_2:.2f}")
            logging.info(f"⛔ Stop Loss: {stop_loss:.2f}")

        elif signal == "SELL" and self.last_entry_price:
            logging.info(f"🔴 SELL order at {price:.2f}, Profit: {price - self.last_entry_price:.2f}")

    def run(self):
        """Menjalankan bot trading"""
        self.train_model()
        
        while True:
            df = self.fetch_data(limit=50)
            df = self.add_indicators(df)
            latest = df.iloc[-1]

            features = ["RSI", "MACD", "MACD_signal", "EMA9", "EMA21", "ATR", "Stoch_K", "Stoch_D"]
            X_latest = self.scaler.transform([latest[features].values])
            prediction = self.model.predict(X_latest)

            logging.info(f"📊 Harga: {latest['close']:.2f}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}")
            
            if prediction == 1:
                logging.info("✅ Model ML memprediksi: BUY")
                self.execute_trade("BUY", latest["close"])
            else:
                logging.info("❌ Model ML memprediksi: Tidak ada aksi")

            time.sleep(60)  # Tunggu 1 menit sebelum analisis ulang
