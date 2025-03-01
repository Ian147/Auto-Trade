import ccxt
binance = ccxt.binance({
    'apiKey': 'j70PupVRg6FbppOVsv0NJeyEYhf24fc9H36XvKQTP496CE8iQpuh0KlurfRGvrLw',
    'secret': 'YGp4SiUdZMQ8ykAszgzSB1eLqv5ZiFM9wZTuV3Z2VOtoM46yDNuy1CBr703PtLVT'
})
balance = binance.fetch_balance()
print(balance)
