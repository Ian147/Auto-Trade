import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

# ✅ Load dataset
DATA_PATH = "data.csv"
df = pd.read_csv(DATA_PATH)

# ✅ Pastikan dataset cukup besar
if len(df) < 250000:
    raise ValueError(f"Dataset hanya memiliki {len(df)} data! Tambahkan lebih banyak data.")

# ✅ Konversi timestamp & pilih fitur
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df[['open', 'high', 'low', 'close', 'volume']]

# ✅ Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# ✅ Simpan scaler
joblib.dump(scaler, "scaler.pkl")

# ✅ Buat dataset untuk LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 3])  # Prediksi harga 'close'
    return np.array(X), np.array(y)

SEQ_LENGTH = 60
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# ✅ Split data
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ✅ Bangun model LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 5)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

# ✅ Register custom loss untuk menghindari error
@tf.keras.utils.register_keras_serializable()
def custom_loss():
    return tf.keras.losses.MeanSquaredError()

model.compile(optimizer="adam", loss=custom_loss())

# ✅ Train model
EPOCHS = 100
BATCH_SIZE = 64

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# ✅ Simpan model
model.save("lstm_model.h5")

print("✅ Model selesai dilatih & disimpan sebagai 'lstm_model.h5'")
