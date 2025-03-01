import ccxt
import time
import pandas as pd
import numpy as np
import ta
import requests

# ✅ KONFIGURASI API BINANCE
api_key = "j70PupVRg6FbppOVsv0NJeyEYhf24fc9H36XvKQTP496CE8iQpuh0KlurfRGvrLw"
api_secret = "YGp4SiUdZMQ8ykAszgzSB1eLqv5ZiFM9wZTuV3Z2VOtoM46yDNuy1CBr703PtLVT"

binance = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"}
})

# ✅ KONFIGURASI TELEGRAM
TELEGRAM_BOT_TOKEN = "8011128170:AAEvCJrvMRinnIsInJmqLjzpWguz88tPWVw"
TELEGRAM_CHAT_ID = "681125756"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")

# ✅ PENGAMBILAN DATA MARKET
def get_data(symbol, timeframe="1h", limit=100):
    bars = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["close"] = df["close"].astype(float)
    return df

# ✅ ANALISIS TEKNIKAL (RSI, MACD, Bollinger Bands)
def analyze_market(symbol):
    df = get_data(symbol)

    # Indikator RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()

    # Indikator MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # **Logika Trading**
    latest = df.iloc[-1]

    if latest["rsi"] < 30 and latest["macd"] > latest["macd_signal"]:
        return "BUY"
    elif latest["rsi"] > 70 and latest["macd"] < latest["macd_signal"]:
        return "SELL"
    return "HOLD"

# ✅ FUNGSI EKSEKUSI ORDER
def place_order(symbol, signal, amount=10):  # Default 10 USDT
    try:
        if signal == "BUY":
            order = binance.create_market_buy_order(symbol, amount / binance.fetch_ticker(symbol)["close"])
        elif signal == "SELL":
            order = binance.create_market_sell_order(symbol, amount / binance.fetch_ticker(symbol)["close"])
        else:
            return None
        
        print(f"Order Executed: {order}")
        send_telegram_message(f"📢 {signal} Order Executed: {order}")
        return order
    except Exception as e:
        print(f"Error placing order: {e}")
        send_telegram_message(f"❌ Order Error: {e}")
        return None

# ✅ FUNGSI TP1, TP2, SL
def manage_trade(symbol, entry_price, tp1_ratio=1.02, tp2_ratio=1.05, sl_ratio=0.98):
    tp1 = entry_price * tp1_ratio
    tp2 = entry_price * tp2_ratio
    sl = entry_price * sl_ratio

    while True:
        current_price = binance.fetch_ticker(symbol)["close"]
        
        if current_price >= tp1:
            send_telegram_message(f"✅ TP1 HIT: {current_price}")
            break
        elif current_price >= tp2:
            send_telegram_message(f"✅ TP2 HIT: {current_price}")
            break
        elif current_price <= sl:
            send_telegram_message(f"🚨 STOP LOSS HIT: {current_price}")
            break

        time.sleep(10)  # Cek harga setiap 10 detik

# ✅ LOOPING UTAMA BOT
symbol = "BTC/USDT"

while True:
    signal = analyze_market(symbol)
    if signal in ["BUY", "SELL"]:
        order = place_order(symbol, signal)
        if order:
            entry_price = order["price"]
            manage_trade(symbol, entry_price)

    print(f"Waiting... ({signal})")
    time.sleep(60)  # Cek market setiap 1 menit
