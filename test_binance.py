import ccxt
import pandas as pd
import numpy as np
import talib
import time
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================== Konfigurasi API ====================
api_key = "6ipqniXiFRmjwGsB8H9vUpgVTexAnsLZ2Ybi0DrLxSKKINMr42wCC8ex7rIrqNlj"
api_secret = "HeINMThVDiJuCaoZFvC16FNj0ZCx9uGs2BxkkS1qTB3PkGTmibXfba3l8DajJ3x0"

binance = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"},
    "rateLimit": 1200
})

# ==================== Fungsi Pengambilan Data ====================
def get_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100):
    bars = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# ==================== Fungsi Perhitungan Indikator ====================
def calculate_indicators(df):
    df["RSI"] = talib.RSI(df["close"], timeperiod=14)

    macd, macdsignal, _ = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_Signal"] = macdsignal

    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2)
    df["Upper_BB"] = upper
    df["Middle_BB"] = middle
    df["Lower_BB"] = lower

    return df

# ==================== Model Machine Learning (XGBoost) ====================
def train_model(data):
    data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)  # 1 = Beli, 0 = Tidak ada aksi
    data.dropna(inplace=True)

    features = ["open", "high", "low", "close", "volume", "RSI", "MACD", "MACD_Signal", "Upper_BB", "Middle_BB", "Lower_BB"]
    X = data[features]
    y = data["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Training Selesai! Akurasi: {accuracy * 100:.2f}%")

    return model, scaler

# ==================== Fungsi Eksekusi Order (Simulasi) ====================
def buy(symbol="BTC/USDT", amount=0.001):
    print(f"🔵 SINYAL BELI: {symbol} sebanyak {amount} BTC (Simulasi)")

def sell(symbol="BTC/USDT", amount=0.001):
    print(f"🔴 SINYAL JUAL: {symbol} sebanyak {amount} BTC (Simulasi)")

# ==================== Strategi Trading ====================
def trade_strategy(symbol="BTC/USDT", model=None, scaler=None):
    df = get_ohlcv(symbol)
    df = calculate_indicators(df)

    latest = df.iloc[-1]
    print(f"Harga saat ini: {latest['close']}, RSI: {latest['RSI']}, MACD: {latest['MACD']}, MACD Signal: {latest['MACD_Signal']}")

    # Prediksi menggunakan model XGBoost
    if model and scaler:
        features = ["open", "high", "low", "close", "volume", "RSI", "MACD", "MACD_Signal", "Upper_BB", "Middle_BB", "Lower_BB"]
        X_latest = scaler.transform([latest[features].values])
        prediction = model.predict(X_latest)[0]

        if prediction == 1:
            print("✅ Model ML memprediksi: BELI")
            buy(symbol)
        else:
            print("❌ Model ML memprediksi: Tidak ada aksi")

# ==================== Loop Utama ====================
def main():
    symbol = "BTC/USDT"
    filename = f"{symbol.replace('/', '_')}_data.csv"

    # Ambil data awal
    data = get_ohlcv(symbol)
    data.to_csv(filename, index=False)
    print(f"✅ Data {symbol} disimpan ke {filename}")

    # Latih model
    model, scaler = train_model(data)

    while True:
        print("\n⏳ Menjalankan strategi trading...")
        trade_strategy(symbol, model, scaler)
        time.sleep(3600)  # Loop setiap 1 jam

# ==================== Jalankan Script ====================
if __name__ == "__main__":
    main()
