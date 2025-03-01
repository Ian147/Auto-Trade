import os
import ccxt
import pandas as pd
import numpy as np
import request
import asyncio
import time
from dotenv import load_dotenv
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from telegram import Bot

# Load API dari file .env
load_dotenv()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Inisialisasi Binance API
exchange = ccxt.binance({
    "apiKey": BINANCE_API_KEY,
    "secret": BINANCE_API_SECRET,
    "options": {"defaultType": "spot"}
})

# Inisialisasi Bot Telegram
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Fungsi untuk mengirim pesan ke Telegram
def send_telegram_message(message):
    asyncio.run(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message))
    
# Fungsi untuk mendapatkan saldo USDT
def get_balance():
    balance = exchange.fetch_balance()
    return balance['total'].get('USDT', 0)

# Fungsi untuk mendapatkan data harga
def fetch_ohlcv(symbol="DOGE/USDT", timeframe="5m", limit=100):
    data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# Fungsi untuk analisis teknikal
def analyze_market(df):
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = EMAIndicator(df["close"], window=200).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    bollinger = BollingerBands(df["close"])
    df["bb_lower"] = bollinger.bollinger_lband()
    df["bb_upper"] = bollinger.bollinger_hband()
    return df

# Fungsi untuk mengecek sentimen berita dari Twitter/X
def get_sentiment():
    url = "https://cryptonews-api.com/api/v1?tickers=DOGE&token=your_news_api_key"
    response = requests.get(url)
    if response.status_code == 200:
        news = response.json()["data"]
        positive = sum(1 for article in news if "bullish" in article["title"].lower())
        negative = sum(1 for article in news if "bearish" in article["title"].lower())
        return "bullish" if positive > negative else "bearish"
    return "neutral"

# Fungsi untuk membuka posisi beli
def place_buy_order(amount=5):
    price = exchange.fetch_ticker("DOGE/USDT")["last"]
    qty = amount / price
    order = exchange.create_market_buy_order("DOGE/USDT", qty)
    send_telegram_message(f"✅ Open BUY Order: {qty:.2f} DOGE @ {price:.4f} USDT\n💰 Balance: {get_balance():.2f} USDT")
    return price

# Fungsi untuk menjual DOGE
def place_sell_order(qty):
    price = exchange.fetch_ticker("DOGE/USDT")["last"]
    order = exchange.create_market_sell_order("DOGE/USDT", qty)
    send_telegram_message(f"📢 SELL Order Executed: {qty:.2f} DOGE @ {price:.4f} USDT\n💰 Balance: {get_balance():.2f} USDT")

# Fungsi utama trading
def trading_bot():
    position = None
    buy_price = 0
    while True:
        df = fetch_ohlcv()
        df = analyze_market(df)
        sentiment = get_sentiment()

        latest = df.iloc[-1]
        price = latest["close"]

        # Syarat Open Posisi BUY
        if position is None and latest["rsi"] < 35 and latest["ema50"] > latest["ema200"] and latest["macd"] > latest["macd_signal"] and latest["close"] <= latest["bb_lower"] and sentiment == "bullish":
            buy_price = place_buy_order()
            position = "long"

        # Syarat Take Profit 1
        elif position == "long" and price >= buy_price * 1.02:
            qty = exchange.fetch_balance()['free']['DOGE'] / 2
            place_sell_order(qty)
            send_telegram_message(f"🎯 TP1 Tercapai: {price:.4f} USDT")

        # Syarat Take Profit 2
        elif position == "long" and price >= buy_price * 1.04:
            qty = exchange.fetch_balance()['free']['DOGE']
            place_sell_order(qty)
            send_telegram_message(f"🏆 TP2 Tercapai: {price:.4f} USDT")
            position = None

        # Stop Loss jika harga turun -1.5%
        elif position == "long" and price <= buy_price * 0.985:
            qty = exchange.fetch_balance()['free']['DOGE']
            place_sell_order(qty)
            send_telegram_message(f"❌ STOP LOSS: {price:.4f} USDT")
            position = None

        time.sleep(60)

# Menjalankan bot
if __name__ == "__main__":
    send_telegram_message("🚀 Bot AI Trading DOGE/USDT dimulai!")
    trading_bot()
