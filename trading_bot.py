import ccxt
import pandas as pd
import numpy as np
import talib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

class TradingBot:
    def __init__(self, symbol="BTC/USDT", timeframe="1h"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = ccxt.binance()
        self.model = None
        self.scaler = None
        self.data = None

    def fetch_data(self, limit=100):
        print("📊 Mengambil data dari Binance...")
        bars = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Tambahkan indikator teknikal
        df["RSI"] = talib.RSI(df["close"], timeperiod=14)
        macd, macdsignal, macdhist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
        df["MACD"] = macd
        df["MACD_Signal"] = macdsignal
        df["Upper_BB"], df["Middle_BB"], df["Lower_BB"] = talib.BBANDS(df["close"], timeperiod=20)

        # Buat target: apakah harga naik atau turun di candle berikutnya
        df["Target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)

        # Hapus nilai NaN
        df.dropna(inplace=True)
        self.data = df
        print("✅ Data berhasil diambil dan diproses.")

    def train_model(self):
        print("📈 Melatih model Machine Learning...")

        features = ["open", "high", "low", "close", "volume", "RSI", "MACD", "MACD_Signal", "Upper_BB", "Middle_BB", "Lower_BB"]
        X = self.data[features]
        y = self.data["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        self.model.fit(X_train, y_train)

        accuracy = self.model.score(X_test, y_test)
        print(f"✅ Model Training Selesai! Akurasi: {accuracy * 100:.2f}%")

    def analyze_market(self):
        """Menganalisis pasar dan memberikan sinyal trading."""
        latest = self.data.iloc[-1]  # Ambil data terbaru
        print(f"📊 Harga: {latest['close']}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, MACD Signal: {latest['MACD_Signal']:.2f}")

        # Persiapkan data terbaru untuk model
        X_latest = pd.DataFrame([latest[["open", "high", "low", "close", "volume", "RSI", "MACD", "MACD_Signal", "Upper_BB", "Middle_BB", "Lower_BB"]]])
        X_latest = self.scaler.transform(X_latest)

        # Prediksi menggunakan model ML
        prediction = self.model.predict(X_latest)[0]

        if prediction == 1:
            print("✅ Model ML memprediksi: BELI")
            self.buy()
        else:
            print("❌ Model ML memprediksi: Tidak ada aksi")

    def buy(self):
        """Menjalankan order beli (dummy, tanpa eksekusi langsung ke Binance)."""
        print(f"🟢 Membeli {self.symbol}")

    def sell(self):
        """Menjalankan order jual (dummy, tanpa eksekusi langsung ke Binance)."""
        print(f"🔴 Menjual {self.symbol}")
