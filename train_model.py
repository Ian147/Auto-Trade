import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from data_fetcher import get_binance_ohlcv  # Pastikan ini tersedia
from config import PAIR

logging.basicConfig(level=logging.INFO)

# Ambil 100.000 data OHLCV dari Binance
def fetch_data():
    df = get_binance_ohlcv(100000)  # Mengambil 100.000 data
    if df is None or df.empty:
        logging.error("❌ Gagal mengambil data dari Binance.")
        exit()
    return df

# Menyiapkan dataset
def prepare_data(df, lookback=50):
    """ Menyiapkan data untuk pelatihan LSTM """
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = df['close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:i+lookback])
        y.append(scaled_data[i+lookback])

    return np.array(X), np.array(y), scaler

# Membuat model LSTM
def build_model(input_shape):
    """ Membangun model LSTM """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Melatih model
def train_model():
    df = fetch_data()
    X, y, scaler = prepare_data(df)
    
    # Bentuk input: (samples, time steps, features)
    model = build_model((X.shape[1], X.shape[2]))

    logging.info("🚀 Mulai melatih model...")
    model.fit(X, y, epochs=20, batch_size=64, validation_split=0.2, verbose=1)

    # Simpan model dan scaler
    model.save("lstm_model.h5")
    np.save("scaler.npy", scaler.scale_)

    logging.info("✅ Model berhasil disimpan sebagai 'lstm_model.h5'")

if __name__ == "__main__":
    train_model()
