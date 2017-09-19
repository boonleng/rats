import os
import argparse
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot
import pandas_datareader
import requests_cache
import chart

# Some global variables
N = 100;                      # Look at stock prices for the last N days
sma_sizes = [10, 50, 100]     # SMA window sizes for moving average
figFolder = 'figs'            # Default folder to save figures

# Some default plotting attributes
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Arial']
matplotlib.rcParams['font.sans-serif'] = ['System Font', 'Verdana', 'Arial']
matplotlib.rcParams['figure.figsize'] = (8, 4)   # Change the size of plots
matplotlib.rcParams['figure.dpi'] = 108

if not os.path.exists(figFolder):
    os.makedirs(figFolder)

def main(args):
    days = N + max(sma_sizes);
    if args.verbose:
        print('Retrieving data for {} for {} days ...'.format(args.symbols, days))

    # End day is always today, then roll back the maximum SMA window, plus another week
    end = datetime.date.today()
    start = end - datetime.timedelta(days = int(days * 1.6))
    #session = requests_cache.CachedSession(cache_name = 'cache', backend = 'sqlite', expire_after = datetime.timedelta(hours = 1))
    stock = pandas_datareader.DataReader(args.symbols, 'yahoo', start, end)

    # Make sure the order of data is ascending
    if stock.major_axis[1] < stock.major_axis[0]:
        print('Reordering data ...')
        # Panel object is usually like this: Dimensions: 5 (items) x days (major_axis) x nstocks (minor_axis)
        stock = stock.sort_index(axis = 1, ascending = True)

    if args.verbose > 1:
        print(stock)
        print(stock.iloc[:, ::-1, 0].head())

    # Google reports at least 250 days, truncate to desired length
    if stock.shape[1] > days:
        if args.verbose > 1:
            print('Truncating data from {} to {} ...'.format(stock.shape[1], days))
        stock = stock.iloc[:, -days:, :]

    # Set up the chart
    view = chart.Chart(N, color_scheme = args.color_scheme)
    view.set_xdata(stock.major_axis[-N:])
    # matplotlib.pyplot.show(block = False)
    # matplotlib.pyplot.show()

    # Go through the symbols
    for symbol in args.symbols:
        view.set_data(stock[:, :, [(symbol)]])
        filename = figFolder + '/' + symbol
        if args.pdf:
            filename = filename + '.pdf'
        else:
            filename = filename + '.png'
        filename = figFolder + '/' + symbol + '.png'
        if args.verbose:
            print('Generating {} ... {} ...'.format(symbol, filename))
        view.savefig(filename)

#
#  M A I N
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'main')
    parser.add_argument('-v', '--verbose', default = 1, action = 'count', help = 'increases verbosity level')
    parser.add_argument('-s', '--symbols', default = ['NVDA', 'TSLA', 'AAPL'], nargs = '+', help = 'specify symbols, e.g., -s NVDA TSLA AAPL')
    parser.add_argument('-c', '--color-scheme', default = 'sunrise', help = 'specify color scheme to use.')
    parser.add_argument('-p', '--pdf', action = 'store_true', help = 'generate PDF.')
    parser.add_argument('-q', '--quiet', action = 'store_true', help = 'quiet mode.')
    args = parser.parse_args()
    if args.quiet:
        args.verbose = 0
    args.symbols = [x.upper() for x in args.symbols]

    # print('symbols = {}'.format(args.symbols))
    try:
        main(args)
    except KeyboardInterrupt:
        print('Exiting ...')
