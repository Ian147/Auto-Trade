import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
import time
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands

# Konfigurasi API Binance
BINANCE_API_KEY = "your_binance_api_key"
BINANCE_SECRET_KEY = "your_binance_secret_key"

binance = ccxt.binance({
    "apiKey": BINANCE_API_KEY,
    "secret": BINANCE_SECRET_KEY,
    "options": {"defaultType": "spot"},
})

# Fungsi mengambil data historis
def fetch_data(symbol="BTC/USDT", timeframe="1h", limit=100):
    bars = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # Indikator teknikal
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    df["ema"] = EMAIndicator(df["close"], window=20).ema_indicator()
    
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    
    return df.dropna()

# Model XGBoost untuk prediksi harga
def train_model():
    df = fetch_data()
    X = df[["rsi", "ema", "bb_upper", "bb_lower"]]
    y = np.where(df["close"].shift(-1) > df["close"], 1, 0)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X, y)
    
    return model

# Fungsi untuk mengambil sinyal trading
def get_signal(model):
    df = fetch_data()
    X = df[["rsi", "ema", "bb_upper", "bb_lower"]].iloc[-1:]
    prediction = model.predict(X)[0]
    
    return "BUY" if prediction == 1 else "SELL"

# Fungsi untuk mengeksekusi order di Binance
def place_order(signal, symbol="BTC/USDT", amount=0.001):
    if signal == "BUY":
        order = binance.create_market_buy_order(symbol, amount)
    elif signal == "SELL":
        order = binance.create_market_sell_order(symbol, amount)
    
    print("Order Executed:", order)

# Main bot loop
if __name__ == "__main__":
    model = train_model()
    
    while True:
        try:
            signal = get_signal(model)
            print(f"Trading Signal: {signal}")
            place_order(signal)
            
            time.sleep(3600)  # Jalankan setiap 1 jam
        except Exception as e:
            print("Error:", e)
            time.sleep(60)
