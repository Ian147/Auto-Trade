from trading_bot import TradingBot

def main():
    bot = TradingBot(symbol="BTCUSDT", timeframe="1h")
    bot.run()

if __name__ == "__main__":
    main()
