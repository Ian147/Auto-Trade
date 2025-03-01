import pandas as pd
import numpy as np
import xgboost as xgb
from binance.client import Client
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Gantilah dengan API Key dan Secret Anda
api_key = '6ipqniXiFRmjwGsB8H9vUpgVTexAnsLZ2Ybi0DrLxSKKINMr42wCC8ex7rIrqNlj'
api_secret = 'HeINMThVDiJuCaoZFvC16FNj0ZCx9uGs2BxkkS1qTB3PkGTmibXfba3l8DajJ3x0'
client = Client(api_key, api_secret)

# Fungsi untuk mengambil data historis BTC/USDT dari Binance
def get_historical_data(symbol='BTCUSDT', interval='1h', lookback='1000'):
    klines = client.get_historical_klines(symbol, interval, lookback + " hours ago UTC")
    data = []
    for kline in klines:
        data.append([
            pd.to_datetime(kline[0], unit='ms'),
            float(kline[1]),  # Open
            float(kline[2]),  # High
            float(kline[3]),  # Low
            float(kline[4]),  # Close
            float(kline[5]),  # Volume
        ])
    df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    return df

# Fungsi untuk memuat dan membersihkan data
def clean_data(df):
    df.dropna(inplace=True)  # Menghapus nilai kosong
    return df

# Fungsi untuk menambah indikator teknikal (Contoh: Simple Moving Average - SMA)
def add_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)  # RSI dengan periode 14
    df = df.dropna()  # Menghapus baris yang kosong karena indikator
    return df

# Fungsi untuk menghitung RSI (Relative Strength Index)
def compute_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fungsi untuk memuat fitur dan label untuk pelatihan
def prepare_data(df):
    X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI']]
    # Menggunakan perbedaan harga tutup untuk label (1 = naik, 0 = turun)
    y = (df['Close'].shift(-1) > df['Close']).astype(int)  # 1 jika harga naik, 0 jika turun
    return X, y

# Fungsi untuk membagi data menjadi training dan testing
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Fungsi untuk melatih model XGBoost
def train_model(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Model XGBoost
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    print("Melatih model...")
    model.fit(X_train, y_train)

    # Evaluasi model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {accuracy * 100:.2f}%")
    return model

# Fungsi untuk memprediksi sinyal perdagangan
def predict_signal(model, X):
    predictions = model.predict(X)
    return predictions

# Fungsi utama untuk bot
def main():
    # Ambil data historis BTC/USDT
    df = get_historical_data(symbol='BTCUSDT', interval='1h', lookback='1000')

    # Bersihkan dan tambah indikator teknikal
    df = clean_data(df)
    df = add_indicators(df)

    # Persiapkan data untuk pelatihan
    X, y = prepare_data(df)

    # Melatih model
    model = train_model(X, y)
    if model:
        print("Model telah berhasil dilatih.")
        
        # Prediksi sinyal perdagangan untuk data terbaru
        new_data = df.iloc[-1:][['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI']]
        
        signal = predict_signal(model, new_data)
        if signal == 1:
            print("Sinyal: Beli")
        elif signal == 0:
            print("Sinyal: Jual")
        else:
            print("Sinyal: Tahan")
    else:
        print("Model gagal dilatih.")

if __name__ == "__main__":
    while True:
        main()
        time.sleep(60 * 60)  # Tunggu 1 jam sebelum mengambil data dan memprediksi lagi
