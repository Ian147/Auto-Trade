import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from config import DATA_PATH, MODEL_PATH, SCALER_PATH

df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

scaler = MinMaxScaler()
df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
np.save(SCALER_PATH, scaler)

X, y = [], []
sequence_length = 60

for i in range(len(df) - sequence_length):
    X.append(df[['open', 'high', 'low', 'close', 'volume']].iloc[i:i+sequence_length].values)
    y.append(df['close'].iloc[i+sequence_length])

X, y = np.array(X), np.array(y)

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 5)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)

model.save(MODEL_PATH)
print("✅ Model disimpan sebagai", MODEL_PATH)
