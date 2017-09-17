import os
import argparse
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot
import pandas_datareader
import requests_cache
import chart
# import joblib
# import multiprocessing

# Some global variables
N = 100;                      # Look at stock prices for the last N days
figFolder = 'figs'            # Default folder to save figures
sma_sizes = [10, 50, 100]     # SMA window sizes

# Some default plotting attributes
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Arial']
matplotlib.rcParams['font.sans-serif'] = ['System Font', 'Verdana', 'Arial']
matplotlib.rcParams['figure.figsize'] = (8, 4)   # Change the size of plots
matplotlib.rcParams['figure.dpi'] = 108

if not os.path.exists(figFolder):
    os.makedirs(figFolder)

def showChart(stock, symbol):
    view = chart.showChart(stock[:, :, symbol], sma_sizes = sma_sizes)
    view['title'] = view['axes'].set_title(symbol)
    view['figure'].savefig(figFolder + '/' + symbol.lower() + '.png')

def main(symbols, verbose = 0):
    max(sma_sizes)

    # num_cores = multiprocessing.cpu_count()
    # print('Number of cores = {}'.format(num_cores))

    if verbose:
        print('Retrieving data for {} ...'.format(symbols))
    # End day is always today, then roll back the maximum SMA window, plus another week
    end = datetime.date.today()
    start = end - datetime.timedelta(days = (N + max(sma_sizes) + 7) * 7 / 5)
    session = requests_cache.CachedSession(cache_name = 'cache', backend = 'sqlite', expire_after = datetime.timedelta(days = 1))
    stock = pandas_datareader.DataReader(symbols, 'yahoo', start, end, session = session)

    for symbol in symbols:
        if verbose:
            print('Generating chart for {}'.format(symbol))
        showChart(stock, symbol)

    # results = joblib.Parallel(n_jobs = num_cores/2)(joblib.delayed(showChart)(sym) for sym in symbols)

#
#  M A I N
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = "main")
    parser.add_argument('-v', '--verbose', default = 0, action = 'count', help = 'increases verbosity level')
    parser.add_argument('-s', '--symbols', default = ['AAPL'], nargs = '+', help = 'specify symbols, e.g., -s AAPL NVDA TSLA')
    args = parser.parse_args()

    # print('Version {}'.format(sys.version_info))

    # print('symbols = {}'.format(args.symbols))
    try:
        main(symbols = args.symbols, verbose = args.verbose)
    except KeyboardInterrupt:
        print('Exiting ...')
