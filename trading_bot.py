import ccxt
import time
import numpy as np
import requests

# Konfigurasi API Binance
api_key = "j70PupVRg6FbppOVsv0NJeyEYhf24fc9H36XvKQTP496CE8iQpuh0KlurfRGvrLw"
api_secret = "YGp4SiUdZMQ8ykAszgzSB1eLqv5ZiFM9wZTuV3Z2VOtoM46yDNuy1CBr703PtLVT"

# Konfigurasi API Telegram
telegram_token = "8011128170:AAEvCJrvMRinnIsInJmqLjzpWguz88tPWVw"
telegram_chat_id = "681125756"

# Inisialisasi Binance
binance = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"}
})

# Pair yang diperdagangkan
symbol = "BTC/USDT"
min_balance = 10   # Minimal saldo USDT
trade_amount = 5   # Order 5 USDT per transaksi
take_profit_pct = 0.02  # +2% dari harga beli
stop_loss_pct = 0.02  # -2% dari harga beli

last_buy_price = None  # Untuk menyimpan harga beli terakhir

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

    return close_prices[-1], ma9, ma21, rsi

# Mengecek saldo USDT
def check_balance():
    balance = binance.fetch_balance()
    usdt_balance = balance['total'].get('USDT', 0)
    return usdt_balance

# Mengecek saldo BTC yang sedang di-hold
def check_btc_balance():
    balance = binance.fetch_balance()
    btc_balance = balance['total'].get('BTC', 0)
    return btc_balance

# Melakukan order BUY atau SELL dan mengirim notifikasi Telegram
def place_order(signal, price):
    global last_buy_price
    usdt_balance = check_balance()

    if usdt_balance < min_balance:
        print("Saldo tidak cukup untuk trading.")
        send_telegram_message("⚠️ *Saldo tidak cukup untuk trading!*")
        return

    order_amount = trade_amount / price  # Open posisi dengan 5 USDT

    if signal == "BUY":
        print(f"Placing BUY order for {order_amount:.6f} BTC at {price:.2f} USDT")
        send_telegram_message(f"📈 *BUY Order Executed*\n🔹 *Amount:* {order_amount:.6f} BTC\n🔹 *Price:* {price:.2f} USDT")
        binance.create_market_buy_order(symbol, order_amount)
        last_buy_price = price  # Simpan harga beli

    elif signal == "SELL":
        btc_balance = check_btc_balance()
        if btc_balance > 0:
            print(f"Placing SELL order for {btc_balance:.6f} BTC at {price:.2f} USDT")
            send_telegram_message(f"📉 *SELL Order Executed*\n🔹 *Amount:* {btc_balance:.6f} BTC\n🔹 *Price:* {price:.2f} USDT")
            binance.create_market_sell_order(symbol, btc_balance)
            last_buy_price = None  # Reset harga beli setelah jual
        else:
            print("Tidak ada BTC untuk dijual.")

# Fungsi utama bot
def trading_bot():
    global last_buy_price
    last_notification_time = 0

    while True:
        try:
            price, ma9, ma21, rsi = get_price_data()
            print(f"Harga: {price:.2f} USDT, MA9: {ma9:.2f}, MA21: {ma21:.2f}, RSI: {rsi:.2f}")

            # **BUY SIGNAL**
            if ma9 > ma21 and rsi < 35:
                print("Sinyal: STRONG BUY")
                send_telegram_message("🟢 *STRONG BUY Signal*")
                place_order("BUY", price)

            # **SELL SIGNAL**
            elif ma9 < ma21 and rsi > 70:
                print("Sinyal: STRONG SELL")
                send_telegram_message("🔴 *STRONG SELL Signal*")
                place_order("SELL", price)

            # **Cek TP & SL**
            if last_buy_price:
                take_profit = last_buy_price * (1 + take_profit_pct)
                stop_loss = last_buy_price * (1 - stop_loss_pct)

                if price >= take_profit:
                    print(f"🔥 Take Profit tercapai di {price:.2f} USDT!")
                    send_telegram_message(f"✅ *Take Profit Tercapai!*\n🔹 *Jual di:* {price:.2f} USDT")
                    place_order("SELL", price)

                elif price <= stop_loss:
                    print(f"⚠️ Stop Loss tercapai di {price:.2f} USDT!")
                    send_telegram_message(f"❌ *Stop Loss Tercapai!*\n🔹 *Jual di:* {price:.2f} USDT")
                    place_order("SELL", price)

            # **Notifikasi Setiap 10 Menit**
            current_time = time.time()
            if current_time - last_notification_time >= 600:  # 600 detik = 10 menit
                usdt_balance = check_balance()
                btc_balance = check_btc_balance()
                send_telegram_message(f"💰 *Saldo USDT:* {usdt_balance:.2f}\n🔹 *BTC dalam Open Posisi:* {btc_balance:.6f}")
                last_notification_time = current_time

            time.sleep(60)  # Tunggu 1 menit sebelum iterasi berikutnya

        except Exception as e:
            print(f"Error: {e}")
            send_telegram_message(f"⚠️ *Error:* {e}")
            time.sleep(10)  # Jika error, tunggu 10 detik sebelum mencoba lagi

# Jalankan bot
trading_bot()
