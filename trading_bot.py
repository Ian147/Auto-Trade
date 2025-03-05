import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from binance.client import Client
from config import *

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
model = tf.keras.models.load_model(MODEL_PATH)
scaler = np.load(SCALER_PATH, allow_pickle=True)

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)

def fetch_latest_data():
    klines = client.get_klines(symbol=PAIR, interval=TIMEFRAME, limit=60)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                       'close_time', 'quote_asset_volume', 'number_of_trades', 
                                       'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return scaler.transform(df.values)

def predict_price():
    X_input = fetch_latest_data()
    X_input = np.expand_dims(X_input, axis=0)
    return model.predict(X_input)[0][0]

def trade():
    prediction = predict_price()
    last_price = float(client.get_symbol_ticker(symbol=PAIR)['price'])
    
    if prediction > last_price * (1 + TP_PERCENT / 100):
        order = client.order_market_buy(symbol=PAIR, quoteOrderQty=TRADE_AMOUNT_USDT)
        send_telegram_message(f"✅ BUY Order: {order}")
    
    elif prediction < last_price * (1 - SL_PERCENT / 100):
        balance = client.get_asset_balance(asset="BTC")
        if float(balance['free']) > 0:
            order = client.order_market_sell(symbol=PAIR, quantity=float(balance['free']))
            send_telegram_message(f"❌ SELL Order: {order}")

while True:
    trade()
    time.sleep(900)
