import logging
from trading_bot import TradingBot

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("🚀 Memulai bot trading...")

    # Inisialisasi bot dengan simbol dan timeframe
    bot = TradingBot(symbol="BTCUSDT", timeframe="1m")
    bot.run()

if __name__ == "__main__":
    main()
