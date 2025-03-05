import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

DATA_PATH = "data.csv"
MODEL_PATH = "lstm_model.h5"

# Baca data
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

# Normalisasi data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[["open", "high", "low", "close", "volume"]])

# Simpan scaler
joblib.dump(scaler, "scaler.pkl")

# Persiapan dataset
SEQ_LEN = 60  # 60 candle terakhir untuk prediksi
X, y = [], []
for i in range(len(scaled_data) - SEQ_LEN):
    X.append(scaled_data[i:i + SEQ_LEN])
    y.append(scaled_data[i + SEQ_LEN, 3])  # Prediksi harga close

X, y = np.array(X), np.array(y)

# Bagi dataset
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# Latih model
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# Simpan model
model.save(MODEL_PATH)
print("✅ Model berhasil disimpan!")
