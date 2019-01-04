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

def today():
    return pandas.to_datetime(datetime.date.today())

def yesterday():
    return today() - pandas.to_timedelta('1 day')

def to_datetime(self):
   return pandas.to_datetime(self)

def file(symbols = None, end = None, days = 330, start = None, folder = 'data', verbose = 0):
    """
        Get a set of stock data from the data folder
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
        local_symbols = [x.upper() for x in symbols]
    else:
        local_symbols = [symbols]
    # Read the first one for the data dimensions
    df = pandas.read_pickle(folder + '/' + local_symbols[0] + '.pkl')
    index = pandas.to_datetime(df.index)
    if end is not None:
        end_datetime = pandas.to_datetime(end)
        if end_datetime > pandas.to_datetime(index[-1]):
            end_datetime = pandas.to_datetime(index[-1])
            if verbose > 1:
                print('Dataset ends at {} ...'.format(end_datetime.strftime('%Y-%m-%d')))
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
            if verbose > 1:
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
    if verbose > 1:
        print('Data indices @ [{}, {}]'.format(start_pos, end_pos))
    df = df.iloc[start_pos:end_pos + 1]
    if verbose:
        print('Loading \033[38;5;198moffline\033[0m data from {} to {} ...'.format(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    # Now we go through the files again and read them this time
    quotes = df.copy()
    for sym in local_symbols[1:]:
        filename = '{}/{}.pkl'.format(folder, sym)
        df = pandas.read_pickle(filename)
        end_pos = index.get_loc(end)
        start_pos = index.get_loc(start)
        df = df.iloc[start_pos:end_pos + 1]
        quotes = pandas.concat([quotes, df], axis = 1)
    #import types
    #quotes.index.to_datetime = types.MethodType(to_datetime, quotes.index)
    quotes.index = pandas.to_datetime(quotes.index)
    if start is None:
        quotes = quotes.iloc[-days:]
    return quotes

def net(symbols, end = today(), days = 130, start = None, engine = 'iex', cache = True, verbose = 0):
    if symbols is None:
        symbols = SYMBOLS
    else:
        if isinstance(symbols, list):
            symbols = [x.upper() for x in symbols]
        else:
            symbols = symbols.upper()
    if start is None:
        start = end - datetime.timedelta(days = int(days * 1.6))
    if cache:
        if verbose:
            print('Loading \033[38;5;46mlive\033[0m data from {} to {} ...'.format(start, end))
        session = requests_cache.CachedSession(cache_name = '.cache-{}'.format(engine),
                                               expire_after = datetime.timedelta(days = 1),
                                               backend = 'sqlite')
        quotes = pandas_datareader.DataReader(symbols, engine, start, end, session = session)
    else:
        if verbose:
            print('Loading \033[38;5;220mlive\033[0m data from {} to {} ...'.format(start, end))
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

def get(symbols, end = today(), days = 130, start = None, folder = 'data', force_net = False):
    """
        Get data
        If the offline folder is present, data will be loaded
        from that folder. Newly added symbols to the script do not
        mean they will be available
    """
    if symbols is None:
        symbols = SYMBOLS
    if isinstance(symbols, list):
        symbols = [x.upper() for x in symbols]
    else:
        symbols = [symbols.upper()]
    all_exist = True
    for symbol in symbols:
        filename = '{}/{}.pkl'.format(folder, symbols[0])
        all_exist &= os.path.exists(filename)
    if all_exist and not force_net:
        quotes = file(symbols, end = end, days = days, start = start, verbose = 1)
    else:
        d = datetime.date.today()
        start = datetime.datetime(d.year - 5, d.month, d.day)
        quotes = net(symbols, end = end, days = days, start = start, verbose = 1)
    return quotes

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

def add_offline(symbols, folder = 'data', verbose = 0):
    """
        add_offline(quotes, symbol)
    """
    if isinstance(symbols, list):
        symbols = [x.upper() for x in symbols]
    else:
        symbols = [symbols.upper()]
    print(symbols)
    for symbol in symbols:
        filename = '{}/{}.pkl'.format(folder, symbol)
        if not os.path.exists(filename):
            d = today()
            start = datetime.datetime(d.year - 5, d.month, d.day)
            quote = net(symbol, start = start)
            # Get around: Somehow pandas Panel data comes in descending order when only one symbol is requested
            if quote.shape[0] > 1 and quote.axes[0][1] < quote.axes[0][0]:
                quote = quote.sort_index(axis = 0, ascending = True)
        else:
            old = file(symbol)
            start = pandas.to_datetime(old.index[-1])
            if start >= pandas.to_datetime('today'):
                continue
            new = net(symbol, start = start)
            u = old.iloc[-1]
            v = new.iloc[0]
            if new.shape[0] is 1 and all(u.sub(v).values < 1.0e-3):
                print('No change in {}, skip adding ...'.format(symbol))
                continue
            quote = pandas.concat([old.iloc[:-1], new])
        if verbose > 1:
            quote.head()
        if verbose:
            print('Saving {} ...'.format(symbol))
        save(quote)
