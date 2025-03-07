import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load Data
df = pd.read_csv("binance_ohlcv_15m.csv")
df["close"] = df["close"].astype(float)

# Cek jika dataset terlalu kecil
if len(df) < 1000:
    raise ValueError("Dataset terlalu kecil. Unduh lebih banyak data dengan `data_fetcher.py`.")

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[["close"]])

# Fungsi untuk menyiapkan dataset LSTM
def create_dataset(data, time_step=50):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Buat dataset
TIME_STEP = 50  # Panjang input sequence
X, y = create_dataset(df_scaled, TIME_STEP)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Bangun Model LSTM
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(TIME_STEP, 1)),
    tf.keras.layers.Dropout(0.2),  # Mencegah overfitting
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Kompilasi Model
model.compile(loss="mse", optimizer="adam")

# Latih Model
model.fit(X, y, epochs=100, batch_size=64, verbose=1, validation_split=0.2)

# Simpan model & scaler
model.save("lstm_model.h5")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model dan scaler berhasil disimpan!")
