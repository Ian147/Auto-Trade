import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Path data dan model
DATA_PATH = "data.csv"  # Sesuaikan dengan dataset yang ada
MODEL_PATH_H5 = "lstm_model.h5"
MODEL_PATH_KERAS = "lstm_model.keras"
SCALER_PATH = "scaler.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Pastikan dataset memiliki cukup data
if len(df) < 1000:
    raise ValueError("Dataset terlalu kecil! Tambahkan lebih banyak data.")

# Konversi timestamp jika belum dalam format datetime
if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Ambil hanya kolom OHLCV
df = df[['open', 'high', 'low', 'close', 'volume']]

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Simpan scaler
joblib.dump(scaler, SCALER_PATH)

# Membentuk data untuk LSTM
SEQ_LEN = 60  # Gunakan data 60 candle sebelumnya untuk prediksi
X, y = [], []

for i in range(SEQ_LEN, len(scaled_data)):
    X.append(scaled_data[i - SEQ_LEN:i])
    y.append(scaled_data[i, 3])  # Prediksi harga close

X, y = np.array(X), np.array(y)

# Split data menjadi train dan validation (80% - 20%)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:], y[train_size:]

# Membuat model LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 5)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Latih model
logging.info("🚀 Mulai melatih model...")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Simpan model
model.save(MODEL_PATH_H5)
model.save(MODEL_PATH_KERAS)
logging.info(f"✅ Model berhasil disimpan sebagai '{MODEL_PATH_H5}' dan '{MODEL_PATH_KERAS}'")
