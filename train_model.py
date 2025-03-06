import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load data
df = pd.read_csv("binance_ohlcv_15m.csv")
df["close"] = df["close"].astype(float)

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
df["close"] = scaler.fit_transform(df[["close"]])

joblib.dump(scaler, "scaler.pkl")  # Simpan scaler

# Membentuk dataset
sequence_length = 50
X, y = [], []

for i in range(len(df) - sequence_length - 1):
    X.append(df["close"].values[i:i+sequence_length])
    y.append(df["close"].values[i+sequence_length])

X, y = np.array(X), np.array(y)

# Split dataset
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Model LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Simpan model
model.save("lstm_model.h5")
print("✅ Model berhasil disimpan!")
