import ccxt
import pandas as pd
import numpy as np
import talib
import time
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Masukkan API Key Binance Testnet
api_key = "6ipqniXiFRmjwGsB8H9vUpgVTexAnsLZ2Ybi0DrLxSKKINMr42wCC8ex7rIrqNlj"
api_secret = "HeINMThVDiJuCaoZFvC16FNj0ZCx9uGs2BxkkS1qTB3PkGTmibXfba3l8DajJ3x0"

# Inisialisasi Binance Testnet
binance = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"},
    "urls": {
        "api": {
            "public": "https://testnet.binance.vision/api",
            "private": "https://testnet.binance.vision/api"
        }
    }
})

# Fungsi untuk mengambil data candlestick
def get_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100):
    bars = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# Hitung indikator teknikal
def calculate_indicators(df):
    df["RSI"] = talib.RSI(df["close"], timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_Signal"] = macdsignal
    df["Upper_BB"], df["Middle_BB"], df["Lower_BB"] = talib.BBANDS(df["close"], timeperiod=20)
    return df

# Fungsi untuk melatih model Machine Learning
def train_model(data):
    data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)  # 1 = Harga naik, 0 = Harga turun
    data.dropna(inplace=True)
    
    features = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'MACD_Signal', 'Upper_BB', 'Middle_BB', 'Lower_BB']
    X = data[features]
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    
    return model

# Fungsi untuk memprediksi menggunakan model ML
def predict(model, df):
    features = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'MACD_Signal', 'Upper_BB', 'Middle_BB', 'Lower_BB']
    latest_data = df[features].iloc[-1:].values
    prediction = model.predict(latest_data)
    return prediction[0]  # 1 = Beli, 0 = Jual

# Fungsi untuk mengeksekusi order beli
def buy(symbol="BTC/USDT", amount=0.001):
    order = binance.create_market_buy_order(symbol, amount)
    print(f"Order beli berhasil: {order}")

# Fungsi untuk mengeksekusi order jual
def sell(symbol="BTC/USDT", amount=0.001):
    order = binance.create_market_sell_order(symbol, amount)
    print(f"Order jual berhasil: {order}")

# Fungsi strategi trading otomatis menggunakan ML
def trade_strategy(model, symbol="BTC/USDT"):
    df = get_ohlcv(symbol)
    df = calculate_indicators(df)

    latest = df.iloc[-1]
    prediction = predict(model, df)

    print(f"Harga saat ini: {latest['close']}, RSI: {latest['RSI']}, MACD: {latest['MACD']}, MACD Signal: {latest['MACD_Signal']}")
    
    if prediction == 1:
        print("==> SINYAL BELI TERDETEKSI")
        buy(symbol)
    else:
        print("==> SINYAL JUAL TERDETEKSI")
        sell(symbol)

# Loop utama untuk menjalankan strategi secara otomatis
def main():
    symbol = "BTC/USDT"
    df = get_ohlcv(symbol, limit=500)
    df = calculate_indicators(df)

    # Simpan data ke CSV
    df.to_csv(f"{symbol.replace('/', '_')}_data.csv", index=False)
    print(f"Data {symbol} berhasil disimpan ke CSV.")

    # Latih model
    model = train_model(df)

    while True:
        print("\nMenjalankan strategi trading...")
        trade_strategy(model, symbol)
        time.sleep(3600)  # Tunggu 1 jam sebelum iterasi berikutnya

# Jalankan bot trading
if __name__ == "__main__":
    main()
