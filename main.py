import logging
from trading_bot import TradingBot

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("🚀 Memulai bot trading...")
    symbol = "BTC/USDT"
    timeframe = "5m"

    bot = TradingBot(symbol, timeframe)
    bot.run()

if __name__ == "__main__":
    main()
