import ccxt
import pandas as pd
import numpy as np
import talib
import time
import logging
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import API_KEY, API_SECRET, SYMBOL, TIMEFRAME

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TradingBot:
    def __init__(self, symbol=SYMBOL, timeframe=TIMEFRAME):
        self.symbol = symbol
        self.timeframe = timeframe
        self.binance = ccxt.binance({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "options": {"defaultType": "spot"},
        })
        self.model = None
        self.scaler = None
        self.features = ["open", "high", "low", "close", "volume", "RSI", "MACD", "MACD_Signal", "Upper_BB", "Middle_BB", "Lower_BB"]

    def fetch_data(self, limit=100):
        """Mengambil data pasar dari Binance."""
        logging.info(f"📊 Mengambil data {self.symbol} dari Binance...")
        bars = self.binance.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Tambahkan indikator teknikal
        df["RSI"] = talib.RSI(df["close"], timeperiod=14)
        macd, macdsignal, _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
        df["MACD"] = macd
        df["MACD_Signal"] = macdsignal
        df["Upper_BB"], df["Middle_BB"], df["Lower_BB"] = talib.BBANDS(df["close"], timeperiod=20)

        # Buat target: 1 jika harga naik, 0 jika turun
        df["Target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)

        # Hapus baris dengan NaN
        df.dropna(inplace=True)
        df.to_csv(f"{self.symbol.replace('/', '_')}_data.csv", index=False)
        logging.info(f"✅ Data {self.symbol} disimpan ke CSV.")
        return df

    def train_model(self, data):
        """Melatih model Machine Learning (XGBoost)."""
        logging.info("📈 Melatih model Machine Learning...")

        X = data[self.features]
        y = data["Target"]

        # Normalisasi data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Pisahkan data untuk training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Inisialisasi dan latih model XGBoost
        self.model = xgb.XGBClassifier(eval_metric="logloss")  # Menghapus use_label_encoder
        self.model.fit(X_train, y_train)

        # Evaluasi akurasi model
        accuracy = self.model.score(X_test, y_test)
        logging.info(f"✅ Model Training Selesai! Akurasi: {accuracy * 100:.2f}%")

    def analyze_market(self):
        """Menganalisis pasar menggunakan model ML dan eksekusi trading."""
        self.data = self.fetch_data(limit=50)
        latest = self.data.iloc[-1]

        logging.info(f"📊 Harga: {latest['close']:.2f}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, MACD Signal: {latest['MACD_Signal']:.2f}")

        # Gunakan model ML untuk prediksi
        X_latest = latest[self.features].values.reshape(1, -1)
        X_latest_scaled = pd.DataFrame(self.scaler.transform(X_latest), columns=self.features)  # Perbaikan warning
        prediction = self.model.predict(X_latest_scaled)[0]

        if prediction == 1:
            logging.info("✅ Model ML memprediksi: BUY SIGNAL")
            self.place_order("buy")
        else:
            logging.info("❌ Model ML memprediksi: Tidak ada aksi")

    def place_order(self, order_type):
        """Menempatkan order di Binance."""
        amount = 0.001  # Bisa disesuaikan
        try:
            if order_type == "buy":
                order = self.binance.create_market_buy_order(self.symbol, amount)
            else:
                order = self.binance.create_market_sell_order(self.symbol, amount)

            logging.info(f"✅ Order {order_type.upper()} berhasil: {order}")
        except Exception as e:
            logging.error(f"❌ Gagal mengeksekusi order: {e}")

    def run(self):
        """Menjalankan bot trading secara otomatis."""
        logging.info("🚀 Memulai bot trading...")

        # Ambil data dan latih model
        self.data = self.fetch_data()
        self.train_model(self.data)

        # Loop analisis pasar
        while True:
            self.analyze_market()
            time.sleep(60)  # Tunggu 1 menit sebelum analisis berikutnya
