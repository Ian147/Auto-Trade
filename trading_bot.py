# trading_bot.py
import time
import pandas as pd
import numpy as np
from binance.client import Client
from ta.momentum import RSIIndicator
from xgboost import XGBClassifier
from config import BINANCE_API_KEY, BINANCE_SECRET_KEY, TRADE_SYMBOL, TIMEFRAME, TRADE_AMOUNT
from telegram_notif import send_telegram_message

# Inisialisasi Binance Client
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Inisialisasi Model AI
model = XGBClassifier()

def get_historical_data(symbol, interval, limit=100):
    """Mengambil data harga historis dari Binance."""
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['close'] = df['close'].astype(float)
    return df

def train_model():
    """Melatih model AI dengan data historis."""
    df = get_historical_data(TRADE_SYMBOL, TIMEFRAME, 200)
    df['RSI'] = RSIIndicator(df['close']).rsi()
    df['label'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # 1 = Buy, 0 = Sell
    X = df[['RSI']].dropna()
    y = df['label'].dropna()
    model.fit(X, y)

def make_decision():
    """Menganalisis data dan menentukan apakah harus Buy atau Sell."""
    df = get_historical_data(TRADE_SYMBOL, TIMEFRAME, 50)
    df['RSI'] = RSIIndicator(df['close']).rsi()
    latest_rsi = df['RSI'].iloc[-1]
    
    # Gunakan AI untuk memprediksi sinyal
    signal = model.predict(np.array([[latest_rsi]]))[0]
    
    if signal == 1:
        return "BUY"
    else:
        return "SELL"

def execute_trade(order_type):
    """Eksekusi order Buy atau Sell."""
    price = float(client.get_symbol_ticker(symbol=TRADE_SYMBOL)['price'])
    tp1 = price * 1.005 if order_type == "BUY" else price * 0.995
    tp2 = price * 1.010 if order_type == "BUY" else price * 0.990
    sl = price * 0.995 if order_type == "BUY" else price * 1.005

    send_telegram_message(f"🔵 Open {order_type} at {price:.2f}, TP1: {tp1:.2f}, TP2: {tp2:.2f}, SL: {sl:.2f}")

    return price, tp1, tp2, sl

def monitor_trade(entry_price, tp1, tp2, sl, order_type):
    """Monitor posisi sampai TP atau SL tercapai."""
    while True:
        price = float(client.get_symbol_ticker(symbol=TRADE_SYMBOL)['price'])

        if order_type == "BUY":
            if price >= tp1:
                send_telegram_message(f"✅ TP1 tercapai di {tp1:.2f}!")
                return
            elif price >= tp2:
                send_telegram_message(f"✅✅ TP2 tercapai di {tp2:.2f}!")
                return
            elif price <= sl:
                send_telegram_message(f"⚠️ Stop Loss tercapai di {sl:.2f}!")
                return
        else:  # SELL
            if price <= tp1:
                send_telegram_message(f"✅ TP1 tercapai di {tp1:.2f}!")
                return
            elif price <= tp2:
                send_telegram_message(f"✅✅ TP2 tercapai di {tp2:.2f}!")
                return
            elif price >= sl:
                send_telegram_message(f"⚠️ Stop Loss tercapai di {sl:.2f}!")
                return
        
        time.sleep(10)  # Cek harga setiap 10 detik

if __name__ == "__main__":
    send_telegram_message("🚀 Bot Trading Binance Dimulai!")
    train_model()

    while True:
        signal = make_decision()
        entry_price, tp1, tp2, sl = execute_trade(signal)
        monitor_trade(entry_price, tp1, tp2, sl, signal)
        time.sleep(60)  # Cek sinyal setiap 1 menit
