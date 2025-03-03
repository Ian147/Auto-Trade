import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from data_fetcher import get_binance_ohlcv
import joblib

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Parameter
LOOKBACK = 50  # Jumlah candle yang digunakan untuk prediksi
EPOCHS = 50    # Jumlah epoch training
BATCH_SIZE = 32

# Ambil data dari Binance
logging.info("📥 Mengambil data OHLCV dari Binance...")
df = get_binance_ohlcv(10000)  # Ambil 10.000 candle

if df is None or df.empty:
    logging.error("❌ Gagal mengambil data OHLCV!")
    exit()

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
df['close_scaled'] = scaler.fit_transform(df['close'].values.reshape(-1, 1))

# Simpan scaler untuk digunakan nanti di bot trading
joblib.dump(scaler, "scaler.pkl")

# Persiapan data untuk LSTM
def prepare_data(df, lookback=LOOKBACK):
    X, y = [], []
    for i in range(len(df) - lookback):
        X.append(df['close_scaled'].iloc[i:i+lookback].values)
        y.append(df['close_scaled'].iloc[i+lookback])

    return np.array(X), np.array(y)

X_train, y_train = prepare_data(df)

# Bentuk input untuk LSTM (samples, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], LOOKBACK, 1)

# Bangun model LSTM
logging.info("🛠️ Membangun model LSTM...")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1)
])

# Kompilasi model
model.compile(optimizer="adam", loss="mean_squared_error")

# Training model
logging.info("🚀 Memulai training model...")
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Simpan model
model.save("lstm_model.h5")
logging.info("✅ Model berhasil disimpan sebagai 'lstm_model.h5'!")
