import sys

MIN_PYTHON = (3, 4)
if sys.version_info < MIN_PYTHON:
    sys.exit('Python %s or later is required.\n' % '.'.join("%s" % n for n in MIN_PYTHON))

import os
import datetime
import pandas
import pandas_datareader
import requests_cache

SYMBOLS = [
    'AAPL', 'TSLA',
    'GOOG', 'BIDU', 'MSFT', 'AABA',
    'NVDA', 'AMAT', 'MRVL', 'MU', 'AMD', 'INTC',
    'AMZN', 'NFLX', 'EBAY', 'BABA', 'BKS',
    'FB', 'TWTR', 'YELP',
    'NDLS', 'CMG', 'MCD',
    'S', 'T', 'VZ', 'TMUS', 'QCOM',
    'SBUX',
    'STX', 'WDC', 'TXN', 'ADI', 'MCHP',
    'SNE',
    'C', 'V', 'BAC', 'WFC', 'AMTD',
    'BP', 'XON', 'CVX', 'OGE',
    'F', 'GM', 'TM'
]
#LATEST_DATE = datetime.date(2017, 9, 23)
#LATEST_DATE = datetime.date(2018, 12, 28)

def to_datetime(self):
   return pandas.to_datetime(self)

def file(symbols = None, end = None, days = 330, start = None, folder = 'data', verbose = 0):
    """
        Get a set of stock data on the selected symbols
        NOTE: If the offline folder is present, data will be loaded
        from that folder. Newly added symbols to the script do not
        mean they will be available
    """
    if not os.path.exists(folder):
        print('Error. Data folder does not exist.')
        return None

    import re
    import glob
    if symbols is None:
        # Gather all the symbols from filenames
        local_symbols = []
        for file in glob.glob(folder + '/*.pkl'):
            m = re.search(folder + '/(.+?).pkl', file)
            if m:
                symbol = m.group(1)
                local_symbols.append(symbol)
    elif isinstance(symbols, list):
        local_symbols = symbols
    else:
        local_symbols = [symbols]
    # Read the first one for the data dimensions
    df = pandas.read_pickle(folder + '/' + local_symbols[0] + '.pkl')
    index = pandas.to_datetime(df.index)
    if end is not None:
        end_datetime = pandas.to_datetime(end)
        # Roll backward to find the previous trading day
        k = 10
        while k > 0 and not index.contains(end_datetime):
            if verbose:
                print('Dataset does not contain {} (not a trading day) (k = {})'.format(end_datetime.strftime('%Y-%m-%d'), k))
            end_datetime -= pandas.to_timedelta('1 day')
            k -= 1
        if not index.contains(end_datetime):
            print('Error. Failed to find end date. Unable to continue.')
            return None
        end_pos = index.get_loc(end_datetime)
        end = df.index[end_pos]
        #df = df.iloc[:end_pos + 1]
    else:
        end = df.index[-1]
        end_pos = df.index.get_loc(end)
    if start is not None:
        start_datetime = pandas.to_datetime(start)
        if start_datetime < pandas.to_datetime(index[0]):
            start_datetime = pandas.to_datetime(index[0])
            if verbose:
                print('Dataset starts at {} ...'.format(start_datetime.strftime('%Y-%m-%d')))
        # Roll forward to find the next trading day
        k = 10
        while k > 0 and not index.contains(start_datetime):
            if verbose:
                print('Dataset does not contain {} (not a trading day) (k = {})'.format(start_datetime.strftime('%Y-%m-%d'), k))
            start_datetime += pandas.to_timedelta('1 day')
            k -= 1
        if not index.contains(start_datetime):
            print('Error. Failed to find start date. Unable to continue.')
            return None
        start_pos = index.get_loc(start_datetime)
        start = df.index[start_pos]
    else:
        start = df.index[0]
        start_pos = index.get_loc(start)
    if verbose:
        print('Data start end indices @ [{}, {}]'.format(start_pos, end_pos))
    df = df.iloc[start_pos:end_pos + 1]
    print('Loading \033[38;5;198moffline\033[0m data from {} to {} ...'.format(start, end))
    # Now we go through the files again and read them this time
    quotes = df.copy()
    for sym in local_symbols[1:]:
        filename = '{}/{}.pkl'.format(folder, sym)
        df = pandas.read_pickle(filename)
        end_pos = index.get_loc(end)
        start_pos = index.get_loc(start)
        df = df.iloc[start_pos:end_pos + 1]
        quotes = pandas.concat([quotes, df], axis = 1)
#    quotes.index.to_datetime = types.MethodType(to_datetime, quotes.index)
    quotes.index = pandas.to_datetime(quotes.index)
    if start is None:
        quotes = quotes.iloc[-days:]
    return quotes

def net(symbols, end = datetime.date.today(), days = 130, start = None, engine = 'iex', cache = False):
    if symbols is None:
        symbols = SYMBOLS
    if start is None:
        start = end - datetime.timedelta(days = int(days * 1.6))
    if cache:
        print('Loading \033[38;5;220mcached\033[0m data from {} to {} ...'.format(start, end))
        session = requests_cache.CachedSession(cache_name = '.data-' + engine + '-cache', backend = 'sqlite', expire_after = datetime.timedelta(days = 5))
        quotes = pandas_datareader.DataReader(symbols, engine, start, end, session = session)
    else:
        print('Loading \033[38;5;46mlive\033[0m data from {} to {} ...'.format(start, end))
        quotes = pandas_datareader.DataReader(symbols, engine, start, end)
    # Make sure time is ascending; Panel dimensions: 5/6 (items) x days (major_axis) x symbols (minor_axis)
    if quotes.shape[0] > 1 and quotes.index[1] < quotes.index[0]:
        quotes = quotes.sort_index(axis = 0, ascending = True)
    if start is None:
        quotes = quotes.iloc[-days:]
    quotes.index = pandas.to_datetime(quotes.index)
    if not isinstance(quotes.columns, pandas.MultiIndex):
        names = ['Attributes', 'Symbols']
        params = quotes.columns.tolist()
        iterables = [params, [symbols]]
        return pandas.DataFrame(quotes.values, index = quotes.index, columns = pandas.MultiIndex.from_product(iterables, names = names))
    return quotes

def data(symbols, end = datetime.date.today(), days = 130, start = None, force_net = False):
    if not force_net:
        quotes = file(symbols, end = end, days = days, start = start)
    else:
        d = datetime.date.today()
        start = datetime.datetime(d.year - 5, d.month, d.day)
        quotes = net(symbols, end = end, days = days, start = start, cache = True)
    quotes.index = pandas.to_datetime(quotes.index)

def save(quotes, folder = 'data'):
    """
        save(quotes)
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    names = list(quotes.columns.names)
    params = quotes.columns.levels[0].tolist()
    symbols = quotes.columns.levels[1].tolist()
    for sym in symbols:
        values = quotes.loc[pandas.IndexSlice[:], (slice(None), sym)].values
        iterables = [params, [sym]]
        df = pandas.DataFrame(values, index = quotes.index, columns = pandas.MultiIndex.from_product(iterables, names = names))
        df.to_pickle(folder + '/' + sym + '.pkl')

def get_frame(quotes, symbol):
    """
        get_frame(quotes, symbol)
    """
    names = list(quotes.columns.names)
    params = quotes.columns.levels[0].tolist()
    iterables = [params, [symbol]]
    values = quotes.loc[pandas.IndexSlice[:], (slice(None), symbol)].values
    return pandas.DataFrame(values, index = quotes.index, columns = pandas.MultiIndex.from_product(iterables, names = names))

def add_offline(symbols, verbose = 0):
    d = datetime.date.today()
    start = datetime.datetime(d.year - 5, d.month, d.day)
    quotes = net(symbols, start = start)
    # Get around: Somehow pandas Panel data comes in descending order when only one symbol is requested
    if quotes.shape[0] > 1 and quotes.axes[0][1] < quotes.axes[0][0]:
        quotes = quotes.sort_index(axis = 0, ascending = True)
    if verbose:
        quotes.head()
    save_to_folder(quotes)
    return quotes
