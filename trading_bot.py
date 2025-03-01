import time
import numpy as np
import pandas as pd
from binance.client import Client
from binance.enums import *
import requests
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import asyncio
from telegram import Bot

# Konfigurasi API Binance (Masukkan API key dan secret dengan aman, jangan hardcode di sini)
API_KEY = "j70PupVRg6FbppOVsv0NJeyEYhf24fc9H36XvKQTP496CE8iQpuh0KlurfRGvrLw"
API_SECRET = "YGp4SiUdZMQ8ykAszgzSB1eLqv5ZiFM9wZTuV3Z2VOtoM46yDNuy1CBr703PtLVT"
client = Client(API_KEY, API_SECRET)

# Konfigurasi Telegram
TELEGRAM_BOT_TOKEN = "8011128170:AAEvCJrvMRinnIsInJmqLjzpWguz88tPWVw"
TELEGRAM_CHAT_ID = "681125756"
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Parameter Trading
PAIR = "BTCUSDT"
TRADE_AMOUNT = 5  # 5 USDT
TP1_PERCENT = 1.5  # 1.5% profit
TP2_PERCENT = 3.0  # 3.0% profit
SL_PERCENT = 1.0   # 1.0% stop loss

# Fungsi untuk mengirim notifikasi ke Telegram
def notify_telegram(message):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message))
    else:
        asyncio.run(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message))

# Kirim notifikasi bahwa bot dimulai
notify_telegram("🤖 Bot AI Trading BTC/USDT dimulai...")

# Fungsi untuk mendapatkan harga historis
def get_price_data():
    try:
        klines = client.get_klines(symbol=PAIR, interval=Client.KLINE_INTERVAL_15MINUTE, limit=50)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                           'close_time', 'quote_asset_volume', 'num_trades', 
                                           'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['close'] = df['close'].astype(float)
        return df
    except Exception as e:
        notify_telegram(f"⚠️ Error mengambil data harga: {e}")
        return None

# Fungsi untuk menganalisis sinyal beli/jual
def get_trading_signal():
    df = get_price_data()
    if df is None:
        return "HOLD"

    # Indikator Teknikal
    df['SMA_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['close'], window=200).sma_indicator()
    df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bb = BollingerBands(df['close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()

    latest = df.iloc[-1]

    if latest['close'] > latest['SMA_50'] and latest['MACD'] > latest['MACD_signal'] and latest['RSI'] < 70 and latest['close'] <= latest['BB_lower']:
        return "BUY"
    elif latest['close'] < latest['SMA_50'] and latest['MACD'] < latest['MACD_signal'] and latest['RSI'] > 30 and latest['close'] >= latest['BB_upper']:
        return "SELL"
    return "HOLD"

# Fungsi untuk cek saldo
def get_balance():
    try:
        balance = client.get_asset_balance(asset='USDT')
        return float(balance['free'])
    except Exception as e:
        notify_telegram(f"⚠️ Error mengambil saldo: {e}")
        return 0.0

# Fungsi untuk menempatkan order
def place_order(order_type, amount):
    try:
        if order_type == "BUY":
            order = client.order_market_buy(symbol=PAIR, quoteOrderQty=amount)
        elif order_type == "SELL":
            order = client.order_market_sell(symbol=PAIR, quoteOrderQty=amount)
        return order
    except Exception as e:
        notify_telegram(f"⚠️ Error menempatkan order {order_type}: {e}")
        return None

# Fungsi utama untuk menjalankan bot
def run_bot():
    position = None
    entry_price = None
    tp1_price = None
    tp2_price = None
    sl_price = None

    while True:
        try:
            balance = get_balance()
            signal = get_trading_signal()
            price = float(client.get_symbol_ticker(symbol=PAIR)['price'])

            if position is None and signal == "BUY":
                order = place_order("BUY", TRADE_AMOUNT)
                if order is not None:
                    entry_price = price
                    tp1_price = entry_price * (1 + TP1_PERCENT / 100)
                    tp2_price = entry_price * (1 + TP2_PERCENT / 100)
                    sl_price = entry_price * (1 - SL_PERCENT / 100)
                    position = "LONG"

                    notify_telegram(f"🚀 Open Posisi BUY BTC/USDT di {entry_price}\n🎯 TP1: {tp1_price}, TP2: {tp2_price}\n⛔ SL: {sl_price}\n💰 Saldo USDT: {balance}")

            elif position == "LONG":
                if price >= tp1_price:
                    notify_telegram(f"✅ TP1 Tercapai di {tp1_price} 🚀")
                if price >= tp2_price:
                    order = place_order("SELL", TRADE_AMOUNT)
                    if order is not None:
                        notify_telegram(f"✅ TP2 Tercapai di {tp2_price}, Close Posisi 🚀")
                        position = None
                if price <= sl_price:
                    order = place_order("SELL", TRADE_AMOUNT)
                    if order is not None:
                        notify_telegram(f"❌ Stop Loss di {sl_price}, Close Posisi 😞")
                        position = None

            time.sleep(60)

        except Exception as e:
            notify_telegram(f"⚠️ Error: {str(e)}")
            time.sleep(60)

# Jalankan bot
if __name__ == "__main__":
    try:
        notify_telegram("🤖 Bot AI Trading BTC/USDT dimulai...")
        run_bot()
    except KeyboardInterrupt:
        notify_telegram("❌ Bot dihentikan secara manual.")
    except Exception as e:
        notify_telegram(f"⚠️ Bot Error: {e}")
