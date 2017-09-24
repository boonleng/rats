import os
import argparse
import datetime
import numpy as np
import data
import chart
import mystyle

# Some global variables
N = 90;                       # Look at stock prices for the last N days
sma_sizes = [10, 50, 100]     # SMA window sizes for moving average
figFolder = 'figs'            # Default folder to save figures
if not os.path.exists(figFolder):
    os.makedirs(figFolder)

def main(args):
    L = N + max(sma_sizes);
    if args.verbose:
        print('Retrieving data for {} for L = {} ...'.format(args.symbols, L))

    # Get the latest data
    stock = data.get_from_net(args.symbols, days = int(L * 1.6))

    # Make sure the order of data is ascending
    if stock.major_axis[1] < stock.major_axis[0]:
        # Panel object is usually like this: Dimensions: 5 (items) x days (major_axis) x nstocks (minor_axis)
        stock = stock.sort_index(axis = 1, ascending = True)

    if args.verbose > 1:
        print(stock)
        print(stock.iloc[:, ::-1, 0].head())

    # Set up the chart
    view = chart.Chart(N, color_scheme = args.color_scheme)
    view.set_xdata(stock.major_axis)

    # Go through the symbols
    for symbol in args.symbols:
        print(stock[:, :, [(symbol)]])
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
