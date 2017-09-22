import os
import datetime
import pandas
import pandas_datareader
import requests_cache

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

def get_old_data(folder = 'data', reload = False):
    """
        Get 5 years worth of data on the selected symbols
    """
    if os.path.exists(folder) and not reload:
        import re
        import glob
        import numpy as np
        # Gather all the symbols from filenames
        local_symbols = []
        for file in glob.glob(folder + '/*.pkl'):
            m = re.search(folder + '/(.+?).pkl', file)
            if m:
                symbol = m.group(1)
                local_symbols.append(symbol)
        # Read the last one for the data dimensions
        df = pandas.read_pickle(file)
        print('Loading offline data from ' + str(df.index[0].strftime('%Y-%m-%d')) + ' to ' + str(df.index[-1].strftime('%Y-%m-%d')) + ' ...')
        #dd = np.empty([df.shape[0], df.shape[1], len(symbols)])
        dd = np.empty([df.shape[1], df.shape[0], len(local_symbols)])
        # Now we go through the files again and read them this time
        for i, sym in enumerate(local_symbols):
            file = folder + '/' + sym + '.pkl'
            df = pandas.read_pickle(file)
            dd[:, :, i] = np.transpose(df.values, (1, 0))
        quotes = pandas.Panel(data = dd, items = df.keys().tolist(), minor_axis = local_symbols, major_axis = df.index.tolist())
    else:
        end = datetime.date(2017, 9, 21)
        start = end - datetime.timedelta(days = 5 * 365)
        print('Loading data from ' + str(start) + ' to ' + str(end) + ' ...')
        session = requests_cache.CachedSession(cache_name = 'cache-big', backend = 'sqlite', expire_after = datetime.timedelta(days = 60))
        quotes = pandas_datareader.DataReader(symbols, 'yahoo', start, end, session = session)
    return quotes

def get_old_indices():
    end = datetime.date(2017, 9, 21)
    start = end - datetime.timedelta(days = 5 * 365)
    print('Loading indices from ' + str(start) + ' to ' + str(end) + ' ...')
    session = requests_cache.CachedSession(cache_name = 'cache-idx', backend = 'sqlite', expire_after = datetime.timedelta(days = 60))
    quotes = pandas_datareader.DataReader(indices, 'yahoo', start, end, session = session)
    return quotes

def get(engine = 'yahoo'):
    end = datetime.date.today()
    start = end - datetime.timedelta(days = 250)
    print('Loading data from ' + str(start) + ' to ' + str(end) + ' ...')
    session = requests_cache.CachedSession(cache_name = 'cache-sub', backend = 'sqlite', expire_after = datetime.timedelta(days = 1))
    quotes = pandas_datareader.DataReader(symbols, engine, start, end, session = session)
    return quotes

def save_to_folder(quotes, folder = 'data'):
    """
        data.save_to_folder(quotes)
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    symbols = quotes.minor_axis.tolist()
    for sym in symbols:
        df = quotes[:, :, sym]
        df.to_pickle(folder + '/' + sym + '.pkl')
