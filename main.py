import time
import logging
from trading_bot import TradingBot

# ✅ Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    symbol = "BTC/USDT"
    timeframe = "1h"
    bot = TradingBot(symbol, timeframe)

    # ✅ Simpan data ke CSV & Latih model
    bot.save_data_to_csv()
    bot.train_model()

    while True:
        bot.trade_strategy()
        time.sleep(3600)  # Tunggu 1 jam sebelum iterasi berikutnya

if __name__ == "__main__":
    main()
