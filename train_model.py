import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import logging
import os

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konstanta
DATA_PATH = "data.csv"  # Path dataset
MODEL_PATH = "lstm_model.h5"  # Path model yang disimpan
SCALER_PATH = "scaler.pkl"  # Path scaler yang disimpan
EPOCHS = 100  # Jumlah epoch pelatihan
BATCH_SIZE = 64  # Batch size untuk training
SEQ_LENGTH = 50  # Jumlah langkah dalam input LSTM
TRAIN_SIZE = 250000  # Gunakan 250k data

# ✅ 1. Muat Data
logger.info("📥 Memuat data...")
df = pd.read_csv(DATA_PATH)

# Pastikan dataset cukup besar
if len(df) < TRAIN_SIZE:
    raise ValueError(f"❌ Dataset hanya memiliki {len(df)} baris, minimal {TRAIN_SIZE} baris diperlukan!")

# Konversi timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Gunakan hanya TRAIN_SIZE data terakhir
df = df.iloc[-TRAIN_SIZE:]

# Pilih fitur yang akan digunakan
features = ["open", "high", "low", "close", "volume"]
data = df[features].values

# ✅ 2. Normalisasi Data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Simpan scaler
joblib.dump(scaler, SCALER_PATH)
logger.info(f"✅ Scaler disimpan: {SCALER_PATH}")

# ✅ 3. Buat Dataset LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 3])  # Target: harga close berikutnya
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Split dataset (80% training, 20% validation)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

logger.info(f"📊 Dataset: Train {X_train.shape}, Validation {X_val.shape}")

# ✅ 4. Bangun Model LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, len(features))),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)  # Prediksi harga close
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# ✅ 5. Latih Model
logger.info("🚀 Mulai melatih model...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

# ✅ 6. Simpan Model
model.save(MODEL_PATH)
logger.info(f"✅ Model berhasil disimpan sebagai '{MODEL_PATH}'")
