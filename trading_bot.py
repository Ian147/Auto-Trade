import ccxt
import pandas as pd
import numpy as np
import talib
import time
import xgboost as xgb
from textblob import TextBlob
import tweepy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Konfigurasi Binance API
binance = ccxt.binance({
    'apiKey': '6ipqniXiFRmjwGsB8H9vUpgVTexAnsLZ2Ybi0DrLxSKKINMr42wCC8ex7rIrqNlj',
    'secret': 'HeINMThVDiJuCaoZFvC16FNj0ZCx9uGs2BxkkS1qTB3PkGTmibXfba3l8DajJ3x0',
    'options': {'defaultType': 'future'}
})

symbol = 'BTC/USDT'
timeframe = '1h'

# === 1. Ambil Data Historis ===
def fetch_data():
    data = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=500)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# === 2. Tambahkan Indikator Teknikal ===
def add_indicators(df):
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
    df['boll_upper'], df['boll_middle'], df['boll_lower'] = talib.BBANDS(df['close'])
    return df.dropna()

# === 3. Analisis Sentimen Twitter ===
def get_sentiment():
    auth = tweepy.OAuthHandler('YOUR_CONSUMER_KEY', 'YOUR_CONSUMER_SECRET')
    auth.set_access_token('YOUR_ACCESS_TOKEN', 'YOUR_ACCESS_SECRET')
    api = tweepy.API(auth)
    tweets = api.search_tweets("Bitcoin", count=50)
    sentiments = [TextBlob(tweet.text).sentiment.polarity for tweet in tweets]
    return sum(sentiments) / len(sentiments) if sentiments else 0

# === 4. Model XGBoost untuk Prediksi ===
def train_xgb_model(df):
    X = df[['rsi', 'macd', 'boll_upper', 'boll_middle', 'boll_lower']]
    y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    
    model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=200)
    model.fit(X[:-1], y[:-1])
    return model

# === 5. Model LSTM untuk Prediksi ===
def train_lstm_model(df):
    X = df[['rsi', 'macd', 'boll_upper', 'boll_middle', 'boll_lower']].values
    y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(1, X.shape[2])),
        LSTM(50),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X[:-1], y[:-1], epochs=50, batch_size=32, verbose=0)
    
    return model

# === 6. Eksekusi Trading ===
def execute_trade(prediction, sentiment, stop_loss_pct=0.02, take_profit_pct=0.04):
    balance = binance.fetch_balance()
    usdt_balance = balance['total']['USDT']
    
    if usdt_balance > 10:
        price = binance.fetch_ticker(symbol)['last']
        amount = (usdt_balance * 0.1) / price
        
        if prediction == 1 and sentiment > 0:
            print("BUY Signal Executed!")
            binance.create_market_buy_order(symbol, amount)
        elif prediction == 0 and sentiment < 0:
            print("SELL Signal Executed!")
            binance.create_market_sell_order(symbol, amount)
        
        # Stop Loss & Take Profit
        stop_loss = price * (1 - stop_loss_pct)
        take_profit = price * (1 + take_profit_pct)
        print(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")

# === 7. Main Loop ===
if __name__ == "__main__":
    df = fetch_data()
    df = add_indicators(df)
    
    # Train Models
    xgb_model = train_xgb_model(df)
    lstm_model = train_lstm_model(df)
    
    while True:
        df = fetch_data()
        df = add_indicators(df)
        
        X_new = df[['rsi', 'macd', 'boll_upper', 'boll_middle', 'boll_lower']].iloc[-1:].values
        X_new_lstm = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))
        
        xgb_pred = xgb_model.predict(X_new)[0]
        lstm_pred = lstm_model.predict(X_new_lstm)[0][0]
        
        sentiment = get_sentiment()
        
        final_prediction = 1 if (xgb_pred + lstm_pred) / 2 > 0.5 else 0
        
        execute_trade(final_prediction, sentiment)
        
        print(f"Prediction: {final_prediction}, Sentiment: {sentiment}")
        
        time.sleep(3600)  # Cek setiap 1 jam
