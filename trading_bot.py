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
    ohlcv = binance.fetch_ohlcv(symbol, timeframe="15m", limit=500)  # Ambil lebih banyak data
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

    df["ma9"] = df["close"].rolling(window=9).mean()
    df["ma21"] = df["close"].rolling(window=21).mean()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["signal"] = 0
    df.loc[(df["ma9"] > df["ma21"]) & (df["rsi"] < 35), "signal"] = 1
    df.loc[(df["ma9"] < df["ma21"]) & (df["rsi"] > 70), "signal"] = -1

    df.dropna(inplace=True)
    return df

# Training Model Machine Learning
def train_ml_model():
    df = get_training_data()
    X = df[["ma9", "ma21", "rsi"]]
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
    price, ma9, ma21, rsi = get_price_data()
    X_input = pd.DataFrame([[ma9, ma21, rsi]], columns=["ma9", "ma21", "rsi"])
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

    delta = np.diff(close_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:])
    avg_loss = np.mean(loss[-14:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    return close_prices[-1], ma9, ma21, rsi

# Mengecek saldo USDT
def check_balance():
    balance = binance.fetch_balance()
    return balance['total'].get('USDT', 0)

# Mengecek saldo BTC yang sedang di-hold
def check_btc_balance():
    balance = binance.fetch_balance()
    return balance['total'].get('BTC', 0)

# Melakukan order BUY atau SELL
def place_order(signal, price):
    usdt_balance = check_balance()

    if usdt_balance < min_balance:
        send_telegram_message("⚠️ *Saldo tidak cukup untuk trading!*")
        return

    order_amount = trade_amount / price

    if signal == "BUY":
        send_telegram_message(f"📈 *BUY Order Executed*\n🔹 *Price:* {price:.2f} USDT")
        binance.create_market_buy_order(symbol, order_amount)

    elif signal == "SELL":
        btc_balance = check_btc_balance()
        if btc_balance > 0:
            send_telegram_message(f"📉 *SELL Order Executed*\n🔹 *Price:* {price:.2f} USDT")
            binance.create_market_sell_order(symbol, btc_balance)

# Fungsi utama bot
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

# Jalankan bot
trading_bot()
