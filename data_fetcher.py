import requests
import pandas as pd
import time

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

def get_binance_ohlcv(symbol="BTCUSDT", interval="15m", limit=1000):
    """
    Mengambil data OHLCV dari Binance.

    :param symbol: Pasangan mata uang, contoh "BTCUSDT"
    :param interval: Timeframe, contoh "15m" (15 menit)
    :param limit: Jumlah data candlestick yang diambil
    :return: DataFrame berisi data OHLCV
    """
    url = f"{BINANCE_API_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    
    try:
        response = requests.get(url)
        data = response.json()

        # Pastikan respons valid
        if isinstance(data, list):
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            
            # Konversi tipe data
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            
            return df
        else:
            print("❌ Gagal mengambil data dari Binance:", data)
            return None
    
    except Exception as e:
        print("⚠️ Error mengambil data Binance:", e)
        return None

# Simpan data
if __name__ == "__main__":
    df = get_binance_ohlcv("BTCUSDT", "15m", 250000)
    if df is not None:
        df.to_csv("data.csv", index=False)
        print(f"✅ Data berhasil disimpan: {len(df)} baris")
