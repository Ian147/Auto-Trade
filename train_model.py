import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Load data dari CSV (pastikan file memiliki 1.000.000 baris data OHLCV)
DATA_PATH = "data.csv"  # Ganti dengan path dataset
df = pd.read_csv(DATA_PATH)

# Pastikan dataset memiliki setidaknya 1.000.000 baris
if len(df) < 1_000_000:
    logging.error("Dataset kurang dari 1.000.000 baris! Tambahkan lebih banyak data.")
    exit()

# Preprocessing Data
df = df[['open', 'high', 'low', 'close', 'volume']]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Simpan scaler untuk normalisasi nanti
joblib.dump(scaler, "scaler.pkl")

# Buat dataset dengan window (timesteps)
SEQUENCE_LENGTH = 60  # Gunakan 60 candle terakhir untuk prediksi
X, y = [], []

for i in range(len(scaled_data) - SEQUENCE_LENGTH):
    X.append(scaled_data[i:i+SEQUENCE_LENGTH])
    y.append(scaled_data[i+SEQUENCE_LENGTH, 3])  # Prediksi harga "close"

X, y = np.array(X), np.array(y)

# Split data menjadi train dan validation set
train_size = int(0.9 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Buat model LSTM
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 5)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# Latih model
EPOCHS = 100  # Gunakan 100 epoch untuk hasil lebih akurat
BATCH_SIZE = 512  # Sesuaikan dengan kapasitas VPS

logging.info("🚀 Mulai melatih model...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    verbose=1
)

# Simpan model
MODEL_PATH = "lstm_model.h5"
model.save(MODEL_PATH)
logging.info(f"✅ Model berhasil disimpan sebagai '{MODEL_PATH}'")
