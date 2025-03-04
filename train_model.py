import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load data
DATA_PATH = "data.csv"
df = pd.read_csv(DATA_PATH)

# Pastikan dataset memiliki setidaknya 250.000 baris
if len(df) < 250000:
    logging.error(f"❌ Dataset hanya memiliki {len(df)} baris, minimal 250.000 diperlukan!")
    exit()

# Konversi timestamp ke datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Gunakan hanya 250.000 data terakhir
df = df.tail(250000)

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[["open", "high", "low", "close", "volume"]])

# Simpan scaler untuk digunakan saat prediksi
np.save("scaler.npy", scaler)

# Persiapan data untuk LSTM
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, 3])  # Target adalah "close"
    return np.array(X), np.array(y)

SEQ_LENGTH = 50  # Gunakan 50 candle terakhir untuk prediksi
X, y = create_sequences(df_scaled, SEQ_LENGTH)

# Split data menjadi training (80%) dan testing (20%)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

logging.info(f"📊 Data Latih: {X_train.shape}, Data Uji: {X_test.shape}")

# Membangun model LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)  # Output satu angka (harga close berikutnya)
])

# Kompilasi model
model.compile(optimizer="adam", loss="mse")

# Melatih model
logging.info("🚀 Mulai melatih model...")
EPOCHS = 100
BATCH_SIZE = 64

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test), verbose=1)

# Simpan model
model.save("lstm_model.h5")
logging.info("✅ Model berhasil disimpan sebagai 'lstm_model.h5'")
