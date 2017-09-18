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

# Some default plotting attributes
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Arial']
matplotlib.rcParams['font.sans-serif'] = ['System Font', 'Verdana', 'Arial']
matplotlib.rcParams['figure.figsize'] = (8, 4)   # Change the size of plots
matplotlib.rcParams['figure.dpi'] = 108

if not os.path.exists(figFolder):
    os.makedirs(figFolder)

def showChart(symbol, stock, color_scheme = 'sunrise'):
    view = chart.showChart(stock[:, :, symbol], color_scheme = color_scheme)
    # view = chart.showChart(stock.iloc[:, :, i], color_scheme = color_scheme)
    view['title'] = view['axes'].set_title(symbol)
    filename = figFolder + '/' + symbol.lower() + '.png'
    view['figure'].savefig(filename)
    matplotlib.pyplot.close(view['figure'])
    os.system('open ' + filename)
    return view

def main(args):
    days = N + 100;
    if args.verbose:
        print('Retrieving data for {} for {} days ...'.format(args.symbols, days))
    # End day is always today, then roll back the maximum SMA window, plus another week
    end = datetime.date.today()
    start = end - datetime.timedelta(days = days)
    session = requests_cache.CachedSession(cache_name = 'cache', backend = 'sqlite', expire_after = datetime.timedelta(days = 1))
    stock = pandas_datareader.DataReader(args.symbols, 'google', start, end, session = session)
    # Google reports at least 250 days, truncate to desired length
    if stock.shape[1] > days:
        if args.verbose > 1:
            print('Truncating data from {} to {} ...'.format(stock.shape[1], days))
        stock = stock.iloc[:, -days:, :]

    # Set up the chart
    view = chart.Chart(N, color_scheme = args.color_scheme)
    view.set_xdata(stock.iloc[:, :, 0].index[-N:])
    # matplotlib.pyplot.show()

    # Go through the symbols
    for symbol in args.symbols:
        if args.verbose:
            print('Generating {} ...'.format(symbol))
        view.set_data(stock[:, :, symbol])
        view.set_title(symbol)
        filename = figFolder + '/' + symbol.lower() + '.png'
        view.savefig(filename)

    # num_cores = multiprocessing.cpu_count()
    # print('Number of cores = {}'.format(num_cores))\
    # with joblib.parallel_backend('threading'):
    #     joblib.Parallel(n_jobs = 2)(joblib.delayed(showChart)(symbol, stock) for symbol in args.symbols)

#
#  M A I N
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = "main")
    parser.add_argument('-v', '--verbose', default = 0, action = 'count', help = 'increases verbosity level')
    parser.add_argument('-s', '--symbols', default = ['AAPL'], nargs = '+', help = 'specify symbols, e.g., -s AAPL NVDA TSLA')
    parser.add_argument('-c', '--color-scheme', default = 'sunrise', help = 'specify color scheme to use.')
    args = parser.parse_args()

    # print('symbols = {}'.format(args.symbols))
    try:
        main(args)
    except KeyboardInterrupt:
        print('Exiting ...')
