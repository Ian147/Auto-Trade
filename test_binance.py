from binance.client import Client
import json

# Load API keys dari file konfigurasi
with open("config.json", "r") as config_file:
    config = json.load(config_file)

client = Client(config["api_key"], config["api_secret"])

# Coba ambil harga terbaru BTCUSDT
price = client.get_symbol_ticker(symbol="BTCUSDT")
print(f"Harga BTC/USDT saat ini: {price['price']}")
