import datetime
import pandas_datareader
import requests_cache

# Some global variables
N = 1350;

# Fang - GOOG, NFLX, AMZN, FB
# Chip - MU, AMAT, MRVL, NVDA
    # '^DJI', '^GSPC', '^IXIC',
indices = ['^DJI', '^GSPC', '^IXIC']
symbols = [
    'AAPL', 'TSLA',
    'GOOG', 'BIDU', 'MSFT',
    'NVDA', 'AMAT', 'MRVL', 'MU', 'AMD', 'INTC',
    'AMZN', 'NFLX', 'EBAY', 'BABA', 'BKS',
    'FB', 'TWTR', 'YELP',
    'NDLS', 'CMG', 'MCD',
    'S', 'T', 'VZ', 'TMUS', 'QCOM',
    'SBUX',
    'STX', 'WDC', 'DVMT', 'TXN', 'ADI', 'MCHP',
    'SNE',
    'C', 'V', 'BAC', 'WFC', 'AMTD',
    'BP', 'XON', 'CVX', 'OGE', 'JASO',
    'F', 'GM', 'TM'
]

# Specify end day, then roll back the maximum SMA window, plus another week
end = datetime.date(2017, 9, 15)
start = end - datetime.timedelta(days = N)

def get_old_data():
    print('Loading data since ' + str(start) + ' ...')
    session = requests_cache.CachedSession(cache_name = 'cache-big', backend = 'sqlite', expire_after = datetime.timedelta(days = 60))
    quotes = pandas_datareader.DataReader(symbols, 'google', start, end, session = session)
    return quotes

def get_old_indices():
    session = requests_cache.CachedSession(cache_name = 'cache-idx', backend = 'sqlite', expire_after = datetime.timedelta(days = 60))
    quotes = pandas_datareader.DataReader(indices, 'yahoo', start, end, session = session)
    return quotes

def get():
    end = datetime.date.today()
    start = end - datetime.timedelta(days = N)
    print('Loading data since from ' + str(start) + ' to ' + str(end) + ' ...')
    session = requests_cache.CachedSession(cache_name = 'cache-sub', backend = 'sqlite', expire_after = datetime.timedelta(days = 1))
    quotes = pandas_datareader.DataReader(symbols, 'yahoo', start, end, session = session)
    return quotes
