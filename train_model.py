import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os

# Load data OHLCV dari file CSV
file_path = "ohlcv_data.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} tidak ditemukan!")

data = pd.read_csv(file_path)

# Ambil hanya 10.000 data terakhir
data = data.tail(10000)

# Preprocessing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume']])

X, y = [], []
time_steps = 60  # Gunakan 60 candle terakhir untuk prediksi
for i in range(len(scaled_data) - time_steps):
    X.append(scaled_data[i:i+time_steps])
    y.append(scaled_data[i+time_steps, 3])  # Prediksi harga close

X, y = np.array(X), np.array(y)

# Split data
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build LSTM Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model dengan 10.000 data
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save Model
model.save("lstm_model.h5")
print("Model telah dilatih dengan 10.000 data dan disimpan sebagai lstm_model.h5")
