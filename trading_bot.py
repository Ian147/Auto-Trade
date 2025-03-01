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
        self.client = Client(api_key="6ipqniXiFRmjwGsB8H9vUpgVTexAnsLZ2Ybi0DrLxSKKINMr42wCC8ex7rIrqNlj", api_secret="HeINMThVDiJuCaoZFvC16FNj0ZCx9uGs2BxkkS1qTB3PkGTmibXfba3l8DajJ3x0")
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
            import requests

# Konfigurasi Telegram
TOKEN = "8011128170:AAEvCJrvMRinnIsInJmqLjzpWguz88tPWVw"  # Ganti dengan Token Bot
CHAT_ID = "681125756"  # Ganti dengan Chat ID

def send_telegram_message(message):
    """Mengirim pesan ke Telegram."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=payload)

# Contoh: Kirim pesan saat Open Posisi
def open_trade(symbol, price, stop_loss, take_profit):
    message = (
        f"📢 *Open Trade*: {symbol}\n"
        f"💰 *Entry Price*: {price}\n"
        f"🛑 *Stop Loss*: {stop_loss}\n"
        f"✅ *Take Profit*: {take_profit}"
    )
    send_telegram_message(message)

# Contoh: Kirim pesan saat trade ditutup (TP/SL tercapai)
def close_trade(symbol, result):
    message = f"🔔 *Trade Closed*: {symbol}\n🚀 *Result*: {result}"
    send_telegram_message(message)

# Contoh penggunaan:
open_trade("BTC/USDT", 84000, 83000, 85000)  # Open posisi
close_trade("BTC/USDT", "Take Profit Hit!")  # TP tercapai
