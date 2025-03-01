import logging
from trading_bot import TradingBot

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("🚀 Memulai bot trading...")
    
    # Atur pasangan koin dan timeframe
    symbol = "BTC/USDT"
    timeframe = "1m"

    bot = TradingBot(symbol, timeframe)
    bot.run()

if __name__ == "__main__":
    main()
