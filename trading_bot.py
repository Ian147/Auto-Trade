import ccxt
import time
import numpy as np
import requests

# 🔹 Konfigurasi API Binance
api_key = "j70PupVRg6FbppOVsv0NJeyEYhf24fc9H36XvKQTP496CE8iQpuh0KlurfRGvrLw"
api_secret = "YGp4SiUdZMQ8ykAszgzSB1eLqv5ZiFM9wZTuV3Z2VOtoM46yDNuy1CBr703PtLVT"

# 🔹 Konfigurasi API Telegram
telegram_token = "8011128170:AAEvCJrvMRinnIsInJmqLjzpWguz88tPWVw"
telegram_chat_id = "681125756"

# 🔹 Inisialisasi Binance
binance = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"}
})

# 🔹 Pair yang diperdagangkan
symbol = "BTC/USDT"
min_balance = 10  # Saldo minimal untuk trading
trade_amount = 5  # USDT yang digunakan dalam 1 order
signal_history = []  # Untuk menghitung akurasi sinyal

# 🔹 Fungsi Kirim Notifikasi ke Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Error mengirim pesan Telegram: {e}")

# 🔹 Mengambil Data Harga, MA, RSI, dan Bollinger Bands
def get_price_data():
    ohlcv = binance.fetch_ohlcv(symbol, timeframe="5m", limit=50)
    close_prices = np.array([x[4] for x in ohlcv])

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

    return close_prices[-1], ma9, ma21, rsi, upper_band, lower_band

# 🔹 Mengecek Saldo USDT
def check_balance():
    balance = binance.fetch_balance()
    usdt_balance = balance['total'].get('USDT', 0)
    return usdt_balance

# 🔹 Mengecek Saldo BTC
def check_btc_balance():
    balance = binance.fetch_balance()
    btc_balance = balance['total'].get('BTC', 0)
    return btc_balance

# 🔹 Melakukan Order BUY atau SELL
def place_order(signal, price):
    usdt_balance = check_balance()
    btc_balance = check_btc_balance()

    if usdt_balance < min_balance and signal == "BUY":
        print("Saldo tidak cukup untuk trading.")
        send_telegram_message("⚠️ *Saldo tidak cukup untuk trading!*")
        return

    order_amount = trade_amount / price  # Open posisi dengan 5 USDT

    if signal == "BUY":
        print(f"Placing BUY order for {order_amount:.6f} BTC at {price:.2f}")
        binance.create_market_buy_order(symbol, order_amount)
        send_telegram_message(f"📈 *BUY Order Executed*\n🔹 *Amount:* {order_amount:.6f} BTC\n🔹 *Price:* {price:.2f} USDT")
        signal_history.append("BUY")

    elif signal == "SELL" and btc_balance > 0:
        print(f"Placing SELL order for {btc_balance:.6f} BTC at {price:.2f}")
        binance.create_market_sell_order(symbol, btc_balance)
        send_telegram_message(f"📉 *SELL Order Executed*\n🔹 *Amount:* {btc_balance:.6f} BTC\n🔹 *Price:* {price:.2f} USDT")
        signal_history.append("SELL")

# 🔹 Menghitung Akurasi Sinyal Trading
def calculate_signal_accuracy():
    total_signals = len(signal_history)
    if total_signals < 2:
        return "Belum cukup data untuk menghitung akurasi."
    
    correct_signals = sum(1 for i in range(1, total_signals) if signal_history[i] != signal_history[i-1])
    accuracy = (correct_signals / (total_signals - 1)) * 100
    return f"📊 *Akurasi Sinyal:* {accuracy:.2f}%"

# 🔹 Fungsi Utama Bot Trading
def trading_bot():
    while True:
        try:
            price, ma9, ma21, rsi, upper_band, lower_band = get_price_data()
            usdt_balance = check_balance()
            btc_balance = check_btc_balance()

            print(f"Harga: {price:.2f}, MA9: {ma9:.2f}, MA21: {ma21:.2f}, RSI: {rsi:.2f}")

            # Kirim update saldo & open posisi
            send_telegram_message(f"💰 *Saldo USDT:* {usdt_balance:.2f} USDT\n🔹 *BTC dalam Open Posisi:* {btc_balance:.6f} BTC")

            # **BUY SIGNAL**
            if ma9 > ma21 and rsi < 35 and price <= lower_band:
                print("Sinyal: STRONG BUY")
                send_telegram_message("🟢 *STRONG BUY Signal*")
                place_order("BUY", price)

            # **SELL SIGNAL**
            elif ma9 < ma21 and rsi > 70 and price >= upper_band:
                print("Sinyal: STRONG SELL")
                send_telegram_message("🔴 *STRONG SELL Signal*")
                place_order("SELL", price)

            else:
                print("Menunggu sinyal trading...")
                send_telegram_message("⌛ *Menunggu sinyal trading...*")

            # Kirim akurasi sinyal ke Telegram setiap 10 transaksi
            if len(signal_history) % 10 == 0:
                send_telegram_message(calculate_signal_accuracy())

            time.sleep(60)  # Tunggu 1 menit sebelum iterasi berikutnya

        except Exception as e:
            print(f"Error: {e}")
            send_telegram_message(f"⚠️ *Error:* {e}")
            time.sleep(10)  # Jika error, tunggu 10 detik sebelum mencoba lagi

# Jalankan bot
trading_bot()
