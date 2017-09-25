import os
import datetime
import pandas
import pandas_datareader
import requests_cache

# Fang - GOOG, NFLX, AMZN, FB
# Chip - MU, AMAT, MRVL, NVDA
    # '^DJI', '^GSPC', '^IXIC',
indices = ['^DJI', '^GSPC', '^IXIC']
SYMBOLS = [
    'AAPL', 'TSLA',
    'GOOG', 'BIDU', 'MSFT', 'AABA',
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
LATEST_DATE = datetime.date(2017, 9, 23)

def get_old_data(folder = 'data', reload = False):
    """
        Get a set of 5-year stock data on the selected symbols
        NOTE: If the offline folder is present, data will be loaded
        from that folder. Newly added symbols to the script do not
        mean they will be available
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
        print('Loading \033[38;5;212moffline\033[0m data from ' + str(df.index[0].strftime('%Y-%m-%d')) + ' to ' + str(df.index[-1].strftime('%Y-%m-%d')) + ' ...')
        dd = np.empty([df.shape[1], df.shape[0], len(local_symbols)])
        # Now we go through the files again and read them this time
        for i, sym in enumerate(local_symbols):
            file = folder + '/' + sym + '.pkl'
            df = pandas.read_pickle(file)
            dd[:, :, i] = np.transpose(df.values, (1, 0))
        quotes = pandas.Panel(data = dd, items = df.keys().tolist(), minor_axis = local_symbols, major_axis = df.index.tolist())
    else:
        quotes = get_from_net(SYMBOLS, end = LATEST_DATE, days = 5 * 365, cache = True)
        save_to_folder(quotes)
    return quotes

def get_old_indices():
    end = LATEST_DATE
    start = end - datetime.timedelta(days = 5 * 365)
    print('Loading indices from ' + str(start) + ' to ' + str(end) + ' ...')
    session = requests_cache.CachedSession(cache_name = 'data-idx-cache', backend = 'sqlite', expire_after = datetime.timedelta(days = 5))
    quotes = pandas_datareader.DataReader(indices, 'yahoo', start, end, session = session)
    return quotes

def get_from_net(symbols, end = datetime.date.today(), days = 150, engine = 'yahoo', cache = False):
    if not isinstance(symbols, list):
        symbols = [symbols]
    start = end - datetime.timedelta(days = days)
    print('Loading data from ' + str(start) + ' to ' + str(end) + ' ...')
    if cache:
        session = requests_cache.CachedSession(cache_name = 'data-' + engine + '-cache', backend = 'sqlite', expire_after = datetime.timedelta(days = 5))
        quotes = pandas_datareader.DataReader(symbols, engine, start, end, session = session)
    else:
        quotes = pandas_datareader.DataReader(symbols, engine, start, end)
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

def add_offline(symbols):
    quotes = get_from_net(symbols, end = LATEST_DATE, days = 5 * 365)
    # Get around: Somehow pandas Panel data comes in descending order when only one symbol is requested
    if quotes.major_axis[0] > quotes.major_axis[1]:
        quotes = quotes.sort_index(axis = 1)
    print(quotes)
    save_to_folder(quotes)
    return quotes
