import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ✅ Load dataset
df = pd.read_csv("data.csv", index_col=0)
df = df[['open', 'high', 'low', 'close', 'volume']]

# ✅ Normalisasi data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
joblib.dump(scaler, "scaler.pkl")

# ✅ Buat dataset untuk LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 3])  # Prediksi harga `close`
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled)
X_train, X_test = X[:-5000], X[-5000:]
y_train, y_test = y[:-5000], y[-5000:]

# ✅ Buat model LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# ✅ Latih model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
model.save("lstm_model.h5")

print("✅ Model LSTM berhasil disimpan: lstm_model.h5")
