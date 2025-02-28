import ccxt
import pandas as pd
import numpy as np
import talib
import time
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Masukkan API Key Binance
api_key = "6ipqniXiFRmjwGsB8H9vUpgVTexAnsLZ2Ybi0DrLxSKKINMr42wCC8ex7rIrqNlj"
api_secret = "HeINMThVDiJuCaoZFvC16FNj0ZCx9uGs2BxkkS1qTB3PkGTmibXfba3l8DajJ3x0"

# Inisialisasi Binance
binance = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"}
})

# Fungsi untuk mengambil data candlestick
def get_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100):
    logging.info(f"📊 Mengambil data {symbol} dari Binance...")
    bars = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# Fungsi untuk menghitung indikator teknikal
def calculate_indicators(df):
    df["RSI"] = talib.RSI(df["close"], timeperiod=14)

    macd, macdsignal, macdhist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_Signal"] = macdsignal

    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df["Upper_BB"] = upper
    df["Middle_BB"] = middle
    df["Lower_BB"] = lower

    return df.dropna()

# Fungsi untuk melatih model ML
def train_model(df):
    logging.info("📈 Melatih model Machine Learning...")

    df = calculate_indicators(df)

    df["Target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)

    features = ["open", "high", "low", "close", "volume", "RSI", "MACD", "MACD_Signal", "Upper_BB", "Middle_BB", "Lower_BB"]
    X = df[features]
    y = df["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    logging.info(f"✅ Model Training Selesai! Akurasi: {accuracy * 100:.2f}%")

    return model, scaler

# Fungsi strategi trading
def trade_strategy(model, scaler, symbol="BTC/USDT"):
    df = get_ohlcv(symbol)
    df = calculate_indicators(df)

    latest = df.iloc[-1]
    latest_features = np.array([latest[["open", "high", "low", "close", "volume", "RSI", "MACD", "MACD_Signal", "Upper_BB", "Middle_BB", "Lower_BB"]]])
    latest_scaled = scaler.transform(latest_features)

    prediction = model.predict(latest_scaled)[0]

    logging.info(f"⏳ Menjalankan strategi trading...")
    logging.info(f"Harga: {latest['close']:.2f}, RSI: {latest['RSI']:.2f}, MACD: {latest['MACD']:.2f}, MACD Signal: {latest['MACD_Signal']:.2f}")

    if prediction == 1:
        logging.info("✅ Model ML memprediksi: SINYAL BELI")
    else:
        logging.info("❌ Model ML memprediksi: Tidak ada aksi")

# Fungsi utama
def main():
    symbol = "BTC/USDT"
    filename = f"{symbol.replace('/', '_')}_data.csv"

    df = get_ohlcv(symbol)
    df.to_csv(filename, index=False)
    logging.info(f"✅ Data {symbol} disimpan ke {filename}")

    model, scaler = train_model(df)

    while True:
        trade_strategy(model, scaler, symbol)
        time.sleep(3600)

if __name__ == "__main__":
    main()
