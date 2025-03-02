import ccxt
import time
import numpy as np
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Konfigurasi API Binance
api_key = "j70PupVRg6FbppOVsv0NJeyEYhf24fc9H36XvKQTP496CE8iQpuh0KlurfRGvrLw"
api_secret = "YGp4SiUdZMQ8ykAszgzSB1eLqv5ZiFM9wZTuV3Z2VOtoM46yDNuy1CBr703PtLVT"

# Konfigurasi API Telegram
telegram_token = "8011128170:AAEvCJrvMRinnIsInJmqLjzpWguz88tPWVw"
telegram_chat_id = "681125756"

# Inisialisasi Binance
binance = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"}
})

# Pair yang diperdagangkan
symbol = "BTC/USDT"
min_balance = 10   # Minimal saldo USDT
trade_amount = 5   # Order 5 USDT per transaksi

# Fungsi Kirim Notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Error mengirim pesan Telegram: {e}")

# Mengambil data harga & indikator untuk Machine Learning
def get_training_data():
    ohlcv = binance.fetch_ohlcv(symbol, timeframe="15m", limit=500)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Moving Averages
    df["ma9"] = df["close"].rolling(window=9).mean()
    df["ma21"] = df["close"].rolling(window=21).mean()

    # MACD
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["signal_macd"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["sma20"] = df["close"].rolling(window=20).mean()
    df["stddev"] = df["close"].rolling(window=20).std()
    df["upper_band"] = df["sma20"] + (df["stddev"] * 2)
    df["lower_band"] = df["sma20"] - (df["stddev"] * 2)

    # ATR (Average True Range)
    df["high-low"] = df["high"] - df["low"]
    df["high-close"] = np.abs(df["high"] - df["close"].shift())
    df["low-close"] = np.abs(df["low"] - df["close"].shift())
    df["true_range"] = df[["high-low", "high-close", "low-close"]].max(axis=1)
    df["atr"] = df["true_range"].rolling(window=14).mean()

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Sinyal Buy & Sell
    df["signal"] = 0
    df.loc[(df["ma9"] > df["ma21"]) & (df["rsi"] < 35) & (df["macd"] > df["signal_macd"]), "signal"] = 1
    df.loc[(df["ma9"] < df["ma21"]) & (df["rsi"] > 70) & (df["macd"] < df["signal_macd"]), "signal"] = -1

    df.dropna(inplace=True)
    return df

# Training Model Machine Learning
def train_ml_model():
    df = get_training_data()
    X = df[["ma9", "ma21", "rsi", "macd", "signal_macd", "upper_band", "lower_band", "atr"]]
    y = df["signal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)
    print(f"🎯 Model Accuracy: {accuracy:.2f}")

    return model, scaler

# Prediksi AI Signal
def predict_signal(model, scaler):
    price, ma9, ma21, rsi, macd, signal_macd, upper_band, lower_band, atr = get_price_data()
    X_input = pd.DataFrame([[ma9, ma21, rsi, macd, signal_macd, upper_band, lower_band, atr]],
                           columns=["ma9", "ma21", "rsi", "macd", "signal_macd", "upper_band", "lower_band", "atr"])
    X_scaled = scaler.transform(X_input)

    prediction = model.predict(X_scaled)[0]

    if prediction == 1:
        print("🔹 Sinyal AI: BUY")
        send_telegram_message("🤖 *AI Signal: BUY* 🚀")
        place_order("BUY", price)
    elif prediction == -1:
        print("🔻 Sinyal AI: SELL")
        send_telegram_message("🤖 *AI Signal: SELL* 📉")
        place_order("SELL", price)
    else:
        print("⏸️ Sinyal AI: HOLD")
        send_telegram_message("🤖 *AI Signal: HOLD* ⏳")

# Mengambil data harga terbaru
def get_price_data():
    ohlcv = binance.fetch_ohlcv(symbol, timeframe="5m", limit=50)
    close_prices = np.array([x[4] for x in ohlcv])

    ma9 = np.mean(close_prices[-9:])
    ma21 = np.mean(close_prices[-21:])

    # MACD
    ema12 = pd.Series(close_prices).ewm(span=12).mean().iloc[-1]
    ema26 = pd.Series(close_prices).ewm(span=26).mean().iloc[-1]
    macd = ema12 - ema26
    signal_macd = pd.Series(macd).ewm(span=9).mean().iloc[-1]

    # Bollinger Bands
    sma20 = np.mean(close_prices[-20:])
    stddev = np.std(close_prices[-20:])
    upper_band = sma20 + (stddev * 2)
    lower_band = sma20 - (stddev * 2)

    # ATR
    atr = np.mean(np.abs(np.diff(close_prices)[-14:]))

    return close_prices[-1], ma9, ma21, rsi, macd, signal_macd, upper_band, lower_band, atr

# Jalankan bot
def trading_bot():
    print("🔄 Training AI Model...")
    model, scaler = train_ml_model()

    while True:
        try:
            predict_signal(model, scaler)
            time.sleep(60)
        except Exception as e:
            send_telegram_message(f"⚠️ *Error:* {e}")
            time.sleep(10)

# Eksekusi bot
trading_bot()
