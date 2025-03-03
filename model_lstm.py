import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from data_fetcher import get_binance_ohlcv

scaler = MinMaxScaler(feature_range=(0, 1))

def prepare_data(df, look_back=50):
    """ Mempersiapkan data untuk LSTM """
    data = df[['close']].values
    data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    
    return np.array(X), np.array(y)

def build_lstm_model():
    """ Membangun model LSTM """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(50, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model():
    """ Melatih model LSTM """
    df = get_binance_ohlcv(5000)  # Ambil 5000 data
    X, y = prepare_data(df)
    
    model = build_lstm_model()
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    model.save("lstm_model.h5")

if __name__ == "__main__":
    train_model()
