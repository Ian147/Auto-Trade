import ccxt

# Masukkan API Key Binance
api_key = "6ipqniXiFRmjwGsB8H9vUpgVTexAnsLZ2Ybi0DrLxSKKINMr42wCC8ex7rIrqNlj"
api_secret = "HeINMThVDiJuCaoZFvC16FNj0ZCx9uGs2BxkkS1qTB3PkGTmibXfba3l8DajJ3x0"

# Inisialisasi Binance
binance = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {"defaultType": "spot"}  # Spot trading
})

# Ambil saldo akun
def cek_saldo():
    saldo = binance.fetch_balance()
    print("Saldo tersedia:", saldo["total"])

# Ambil harga pasar real-time
def harga_market(symbol="BTC/USDT"):
    ticker = binance.fetch_ticker(symbol)
    print(f"Harga {symbol} saat ini: {ticker['last']} USDT")

# Order beli
def beli(symbol="BTC/USDT", amount=0.001):
    order = binance.create_market_buy_order(symbol, amount)
    print("Order beli berhasil:", order)

# Order jual
def jual(symbol="BTC/USDT", amount=0.001):
    order = binance.create_market_sell_order(symbol, amount)
    print("Order jual berhasil:", order)

# Contoh penggunaan
cek_saldo()
harga_market()
