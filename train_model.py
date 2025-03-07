import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load Data
df = pd.read_csv("BTCUSDT_100k.csv", index_col=0)
df["close"] = df["close"].astype(float)

# Normalisasi Data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df["close"].values.reshape(-1, 1))

# Buat Data untuk LSTM
lookback = 50  # Gunakan 50 candle terakhir
X, y = [], []
for i in range(len(data_scaled) - lookback):
    X.append(data_scaled[i:i + lookback])
    y.append(data_scaled[i + lookback])

X, y = np.array(X), np.array(y)

# Bangun Model LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
print("✅ Model LSTM Dibangun!")

# Latih Model
model.fit(X, y, epochs=50, batch_size=64, validation_split=0.2)

# Simpan Model
model.save("lstm_model.h5")
print("✅ Model berhasil disimpan!")
