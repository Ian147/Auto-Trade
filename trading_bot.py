import time
import numpy as np
import pandas as pd
import logging
from binance.client import Client
from keras.models import load_model

# Konfigurasi API Binance
API_KEY = "ISI_DENGAN_API_KEY"
API_SECRET = "ISI_DENGAN_API_SECRET"

client = Client(API_KEY, API_SECRET)

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Muat model LSTM yang sudah dilatih
try:
    model = load_model("lstm_model.h5")
    logger.info("✅ Model LSTM berhasil dimuat!")
except Exception as e:
    logger.error(f"❌ Gagal memuat model: {e}")
    exit()

# Fungsi mendapatkan data OHLCV dari Binance
def get_binance_data(symbol="BTCUSDT", interval="15m", limit=1000):
    """
    Mengambil data historis dari Binance dalam bentuk DataFrame.
    """
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                           'close_time', 'quote_asset_volume', 'number_of_trades',
                                           'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['close'] = df['close'].astype(float)  # Pastikan harga closing float
        return df[['close']]
    except Exception as e:
        logger.error(f"❌ Error mengambil data dari Binance: {e}")
        return None

# Fungsi menyiapkan data untuk model LSTM
def prepare_data(data, lookback=20):
    """
    Mengubah data harga closing menjadi format yang sesuai untuk LSTM.
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    
    return np.array(X), np.array(y)

# Fungsi untuk melakukan prediksi harga dan sinyal trading
def predict_signal():
    """
    Mengambil data dari Binance, memproses, dan membuat prediksi dengan LSTM.
    """
    df = get_binance_data()
    if df is None or df.empty:
        return None
    
    lookback = 20  # Jumlah candle yang digunakan untuk prediksi
    X_test, _ = prepare_data(df['close'].values, lookback)

    if len(X_test) == 0:
        logger.warning("⚠️ Data tidak cukup untuk prediksi!")
        return None

    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    predicted_price = model.predict(X_test[-1].reshape(1, lookback, 1))
    
    last_close = df['close'].iloc[-1]
    if predicted_price > last_close:
        return "BUY"
    elif predicted_price < last_close:
        return "SELL"
    else:
        return "HOLD"

# Fungsi eksekusi order di Binance
def execute_trade(signal, amount=10):
    """
    Menjalankan market order berdasarkan sinyal yang diberikan.
    """
    try:
        if signal == "BUY":
            order = client.order_market_buy(symbol="BTCUSDT", quoteOrderQty=amount)
            logger.info(f"✅ Order BUY berhasil: {order}")
        elif signal == "SELL":
            balance = client.get_asset_balance(asset="BTC")
            btc_amount = float(balance['free'])
            if btc_amount > 0:
                order = client.order_market_sell(symbol="BTCUSDT", quantity=btc_amount)
                logger.info(f"✅ Order SELL berhasil: {order}")
            else:
                logger.warning("⚠️ Tidak ada saldo BTC untuk dijual.")
        else:
            logger.info("⚠️ Tidak ada aksi trading.")
    except Exception as e:
        logger.error(f"❌ Gagal eksekusi trade: {e}")

# Fungsi utama bot trading
def trading_bot():
    """
    Loop utama bot trading untuk terus membaca data dan eksekusi order.
    """
    while True:
        try:
            signal = predict_signal()
            if signal:
                logger.info(f"📊 Sinyal: {signal}")
                execute_trade(signal)
            time.sleep(60)  # Cek setiap 1 menit
        except KeyboardInterrupt:
            logger.info("🚪 Bot dihentikan oleh pengguna.")
            break
        except Exception as e:
            logger.error(f"❌ Error dalam trading bot: {e}")
            time.sleep(60)

# Jalankan bot
if __name__ == "__main__":
    trading_bot()
