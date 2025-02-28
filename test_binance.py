import ccxt
import pandas as pd
import numpy as np
import talib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

class TradingBot:
    def __init__(self, symbol="BTC/USDT", timeframe="1h"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.api_key = "6ipqniXiFRmjwGsB8H9vUpgVTexAnsLZ2Ybi0DrLxSKKINMr42wCC8ex7rIrqNlj"
        self.api_secret = "HeINMThVDiJuCaoZFvC16FNj0ZCx9uGs2BxkkS1qTB3PkGTmibXfba3l8DajJ3x0"

        # ✅ Gunakan akun real Binance
        self.exchange = ccxt.binance({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "options": {"defaultType": "spot"}
        })

        self.model = None
        self.scaler = None

    # ✅ Ambil data candlestick
    def get_ohlcv(self):
        logging.info(f"📊 Mengambil data {self.symbol} dari Binance...")
        bars = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=100)
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    # ✅ Hitung indikator teknikal
    def calculate_indicators(self, df):
        df["RSI"] = talib.RSI(df["close"], timeperiod=14)
        macd, macdsignal, _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
        df["MACD"] = macd
        df["MACD_Signal"] = macdsignal
        upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20)
        df["Upper_BB"] = upper
        df["Middle_BB"] = middle
        df["Lower_BB"] = lower
        df.dropna(inplace=True)
        return df

    # ✅ Simpan data ke CSV
    def save_data_to_csv(self):
        df = self.get_ohlcv()
        filename = f"{self.symbol.replace('/', '_')}_data.csv"
        df.to_csv(filename, index=False)
        logging.info(f"✅ Data {self.symbol} disimpan ke {filename}")

    # ✅ Latih model Machine Learning
    def train_model(self):
        logging.info("📈 Melatih model Machine Learning...")
        df = self.get_ohlcv()
        df = self.calculate_indicators(df)
        df["Target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)

        features = ["open", "high", "low", "close", "volume", "RSI", "MACD", "MACD_Signal", "Upper_BB", "Middle_BB", "Lower_BB"]
        X = df[features]
        y = df["Target"]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.model = xgb.XGBClassifier()
        self.model.fit(X_train, y_train)

        accuracy = self.model.score(X_test, y_test)
        logging.info(f"✅ Model Training Selesai! Akurasi: {accuracy * 100:.2f}%")

    # ✅ Eksekusi order beli
    def buy(self, amount=0.001):
        logging.info(f"🟢 SINYAL BELI TERDETEKSI! Membeli {amount} {self.symbol}")
        order = self.exchange.create_market_buy_order(self.symbol, amount)
        logging.info(f"✅ Order beli berhasil: {order}")

    # ✅ Eksekusi order jual
    def sell(self, amount=0.001):
        logging.info(f"🔴 SINYAL JUAL TERDETEKSI! Menjual {amount} {self.symbol}")
        order = self.exchange.create_market_sell_order(self.symbol, amount)
        logging.info(f"✅ Order jual berhasil: {order}")

    # ✅ Strategi trading
    def trade_strategy(self):
        df = self.get_ohlcv()
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]

        logging.info(f"⏳ Menjalankan strategi trading...")
        logging.info(f"Harga: {latest['close']}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, MACD Signal: {latest['MACD_Signal']:.2f}")

        features = ["open", "high", "low", "close", "volume", "RSI", "MACD", "MACD_Signal", "Upper_BB", "Middle_BB", "Lower_BB"]
        latest_features = np.array([latest[features]])
        latest_scaled = self.scaler.transform(latest_features)

        prediction = self.model.predict(latest_scaled)[0]
        
        if prediction == 1:
            self.buy()
        else:
            logging.info("❌ Model ML memprediksi: Tidak ada aksi")
