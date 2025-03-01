import logging
import pandas as pd
import numpy as np
import talib
import xgboost as xgb
from binance.client import Client
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

class TradingBot:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.client = Client(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET")
        self.model = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.03, use_label_encoder=False, eval_metric="logloss")
        self.scaler = StandardScaler()
        self.data = None
        self.tp_multiplier = 1.02   # Take Profit 2%
        self.sl_multiplier = 0.98   # Stop Loss 2%
    
    def fetch_data(self, limit=5000):
        klines = self.client.get_klines(symbol=self.symbol, interval=self.timeframe, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', '_', '_', '_', '_', '_', '_'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Tambahkan indikator teknikal yang lebih banyak
        df["rsi"] = talib.RSI(df["close"], timeperiod=14)
        df["macd"], df["macd_signal"], _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
        df["ema50"] = talib.EMA(df["close"], timeperiod=50)
        df["ema200"] = talib.EMA(df["close"], timeperiod=200)
        df["stochastic"] = talib.STOCH(df["high"], df["low"], df["close"])[0]
        df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
        df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
        upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        df["volatility"] = (df["high"] - df["low"]) / df["close"]
        df["volume_change"] = df["volume"].pct_change()

        df.dropna(inplace=True)
        return df

    def train_model(self):
        self.data = self.fetch_data()
        X = self.data[['rsi', 'macd', 'macd_signal', 'ema50', 'ema200', 'stochastic', 'adx', 'atr', 'bb_upper', 'bb_lower', 'volatility', 'volume_change']]
        y = (self.data['close'].shift(-1) > self.data['close']).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        accuracy = cross_val_score(self.model, X_test_scaled, y_test, cv=5).mean()
        logging.info(f"✅ Model Training Selesai! Akurasi: {accuracy:.2%}")

    def predict(self, latest_data):
        X_new = latest_data[['rsi', 'macd', 'macd_signal', 'ema50', 'ema200', 'stochastic', 'adx', 'atr', 'bb_upper', 'bb_lower', 'volatility', 'volume_change']].values.reshape(1, -1)
        X_new_scaled = self.scaler.transform(X_new)
        prediction = self.model.predict(X_new_scaled)[0]
        return prediction

    def execute_trade(self, price, action, atr_value):
        if action == "BUY":
            tp_price = price * self.tp_multiplier  # Take Profit 2%
            sl_price = price - (atr_value * 1.5)  # ATR-based Stop Loss

            logging.info(f"✅ ORDER BUY @ {price:.2f} | TP: {tp_price:.2f}, SL: {sl_price:.2f}")
        elif action == "SELL":
            logging.info(f"✅ ORDER SELL @ {price:.2f}")

    def run(self):
        self.train_model()
        while True:
            df = self.fetch_data(limit=50)
            latest = df.iloc[-1]
            action = self.predict(latest)

            logging.info(f"📊 Harga: {latest['close']:.2f}, RSI: {latest['rsi']:.2f}, MACD: {latest['macd']:.2f}, ATR: {latest['atr']:.2f}, ADX: {latest['adx']:.2f}")

            if action == 1:
                self.execute_trade(latest["close"], "BUY", latest["atr"])
            else:
                logging.info("❌ Model ML memprediksi: Tidak ada aksi")
