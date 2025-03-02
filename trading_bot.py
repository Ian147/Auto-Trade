import ccxt
import time
import numpy as np
import requests

# Konfigurasi API Binance
api_key = "MASUKKAN_API_KEY_BINANCE"
api_secret = "MASUKKAN_SECRET_KEY_BINANCE"

# Konfigurasi API Telegram
telegram_token = "MASUKKAN_TOKEN_BOT_TELEGRAM"
telegram_chat_id = "MASUKKAN_CHAT_ID_TELEGRAM"

# Inisialisasi Binance
binance = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"}
})

# Pair yang diperdagangkan
symbol = "DOGE/USDT"
min_balance = 13  # Saldo minimal untuk trading
trade_amount = 5  # USDT yang digunakan dalam 1 order (DIPERBARUI ke 5 USDT)

# Fungsi Kirim Notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Error mengirim pesan Telegram: {e}")

# Mengambil data harga, MA, RSI, dan Bollinger Bands
def get_price_data():
    ohlcv = binance.fetch_ohlcv(symbol, timeframe="5m", limit=50)
    close_prices = np.array([x[4] for x in ohlcv])
    volumes = np.array([x[5] for x in ohlcv])

    # Moving Averages
    ma9 = np.mean(close_prices[-9:])
    ma21 = np.mean(close_prices[-21:])

    # RSI Calculation
    delta = np.diff(close_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:])
    avg_loss = np.mean(loss[-14:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Bollinger Bands
    sma20 = np.mean(close_prices[-20:])
    std_dev = np.std(close_prices[-20:])
    upper_band = sma20 + (2 * std_dev)
    lower_band = sma20 - (2 * std_dev)

    # Volume Analysis
    avg_volume = np.mean(volumes[-10:])
    last_volume = volumes[-1]
    volume_signal = "HIGH" if last_volume > avg_volume else "LOW"

    return close_prices[-1], ma9, ma21, rsi, upper_band, lower_band, volume_signal

# Mengecek saldo USDT
def check_balance():
    balance = binance.fetch_balance()
    usdt_balance = balance['total'].get('USDT', 0)
    return usdt_balance

# Melakukan order BUY atau SELL dan mengirim notifikasi Telegram
def place_order(signal, price):
    usdt_balance = check_balance()

    if usdt_balance < min_balance:
        print("Saldo tidak cukup untuk trading.")
        send_telegram_message("⚠️ *Saldo tidak cukup untuk trading!*")
        return

    order_amount = trade_amount / price  # Open posisi dengan 5 USDT

    if signal == "BUY":
        print(f"Placing BUY order for {order_amount:.2f} DOGE at {price:.4f}")
        send_telegram_message(f"📈 *BUY Order Executed*\n🔹 *Amount:* {order_amount:.2f} DOGE\n🔹 *Price:* {price:.4f} USDT")
        binance.create_market_buy_order(symbol, order_amount)

    elif signal == "SELL":
        balance = binance.fetch_balance()
        doge_balance = balance['total'].get('DOGE', 0)
        if doge_balance > 0:
            print(f"Placing SELL order for {doge_balance:.2f} DOGE at {price:.4f}")
            send_telegram_message(f"📉 *SELL Order Executed*\n🔹 *Amount:* {doge_balance:.2f} DOGE\n🔹 *Price:* {price:.4f} USDT")
            binance.create_market_sell_order(symbol, doge_balance)
        else:
            print("Tidak ada DOGE untuk dijual.")

# Fungsi utama bot
def trading_bot():
    while True:
        try:
            price, ma9, ma21, rsi, upper_band, lower_band, volume_signal = get_price_data()
            print(f"Harga: {price:.4f}, MA9: {ma9:.4f}, MA21: {ma21:.4f}, RSI: {rsi:.2f}, Vol: {volume_signal}")

            # Kirim status saldo dan open posisi
            usdt_balance = check_balance()
            open_positions = binance.fetch_balance()['total'].get('DOGE', 0)
            send_telegram_message(f"💰 *Saldo USDT:* {usdt_balance:.2f}\n🔹 *DOGE dalam Open Posisi:* {open_positions:.2f}")

            # **BUY SIGNAL**
            if ma9 > ma21 and rsi < 35 and price <= lower_band and volume_signal == "HIGH":
                print("Sinyal: STRONG BUY")
                send_telegram_message("🟢 *STRONG BUY Signal*")
                place_order("BUY", price)

            # **SELL SIGNAL**
            elif ma9 < ma21 and rsi > 70 and price >= upper_band and volume_signal == "HIGH":
                print("Sinyal: STRONG SELL")
                send_telegram_message("🔴 *STRONG SELL Signal*")
                place_order("SELL", price)

            else:
                print("Menunggu sinyal trading...")
                send_telegram_message("⌛ *Menunggu sinyal trading...*")

            time.sleep(60)  # Tunggu 1 menit sebelum iterasi berikutnya

        except Exception as e:
            print(f"Error: {e}")
            send_telegram_message(f"⚠️ *Error:* {e}")
            time.sleep(10)  # Jika error, tunggu 10 detik sebelum mencoba lagi

# Jalankan bot
trading_bot()
