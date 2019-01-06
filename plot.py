#!/usr/bin/env python3

"""
@author: Boonleng Cheong
"""

import sys

MIN_PYTHON = (3, 4)
if sys.version_info < MIN_PYTHON:
    sys.exit('Python %s or later is required.\n' % '.'.join("%s" % n for n in MIN_PYTHON))

import os
import multiprocessing
import argparse
import data
import chart
import time
from ast import literal_eval

def savefig(chart, data_frame, filename):
    chart.set_data(data_frame)
    chart.savefig(filename)

def genfigs(symbols, days = 165, end = None, sma_sizes = chart.DEFAULT_SMA_SIZES, folder = 'figs',
            color_scheme = 'default', image_format = 'png', dpi = 144, verbose = 0,
            open_preview = False, force_net = False, figsize = (8.89, 5.0)):
    # Get the latest data
    if symbols == '^OLD':
        stock = data.file(end = end)
        symbols = list(stock.columns.levels[1])
    elif force_net is False:
        stock = data.file(symbols = symbols, end = end, verbose = verbose)
    else:
        # Total data length to retrieve to have complete valid SMA
        L = days + max(sma_sizes);
        if verbose:
            print('Retrieving data for {} for L = {} ...'.format(symbols, L))
        stock = data.net(symbols, days = int(L * 1.6))

    # Show parts of the data if we are in verbose mode
    if verbose:
        print('symbols = {} ({})   end = {}'.format(symbols, len(symbols), end))

    # Get the core count
    batch_size = min(multiprocessing.cpu_count(), len(symbols))
    
    # Set up the chart
    print('Preparing background (batch size = {}) ...'.format(batch_size))
    views = []
    for _ in range(batch_size):
        view = chart.Chart(n = days, color_scheme = color_scheme, figsize = figsize, dpi = int(dpi))
        views.append(view)

    # Create the output folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Go through the symbols
    t1 = time.time()
    procs = []
    command = 'open'
    if len(symbols) > 1:
        symbols = stock.columns.levels[1].tolist()
    for i, symbol in enumerate(symbols):
        if len(symbols) > 1:
            data_frame = data.get_frame(stock, symbol)
        else:
            data_frame = stock
        if verbose > 1:
            print(data_frame)
        # Derive the filename
        filename = '{}/{}.{}'.format(folder, symbol, image_format)
        oc = data_frame.loc[:, (['open', 'close'])].values[-1]
        delta = oc[1] - oc[0]
        if delta > 0:
            print('\033[38;5;46m{}\033[0m -> \033[38;5;206m{}\033[0m'.format(symbol.rjust(5), filename))
        else:
            print('\033[38;5;196m{}\033[0m -> \033[38;5;206m{}\033[0m'.format(symbol.rjust(5), filename))
       # Update data and save an image
        proc = multiprocessing.Process(target = savefig, args = (views[i % batch_size], data_frame, filename))
        procs.append(proc)
        proc.start()
        # Wait for all of them to finish
        if len(procs) == batch_size:
            for proc in procs:
                proc.join()
            procs = []
        command += ' ' + filename

    # Finish whatever that is left in the queue
    if len(procs) > 0:
        for proc in procs:
            proc.join()

    t0 = time.time()
    
    if verbose > 1:
        print('Elapsed time: {0:.3f} s'.format(t0 - t1))

    if open_preview:
        os.system(command)


#
#  M A I N
#
if __name__ == '__main__':
    usage = '''
    python plot.py [options] SYMBOLS

    examples:
    
    python plot.py NVDA
    python plot.py FB AMZN
    python plot.py -s AAPL
    python plot.py -x TSLA
    python plot.py -n -o GOOG MSFT
    python plot.py -e 2017-09-25 -s AAPL
    python plot.py -e 2017-12-31 NVDA TSLA GOOG
    '''
    parser = argparse.ArgumentParser(prog = 'plot', usage = usage)
    parser.add_argument('symbols', default = '^OLD', nargs = '*', help = 'specify symbols, e.g., NVDA TSLA AAPL')
    parser.add_argument('-c', '--color-scheme', default = 'default', help = 'specify color scheme to use (sunrise, sunset, night)')
    parser.add_argument('-d', '--days', default = 165, help = 'specify the number of days')
    parser.add_argument('-e', '--end', default = None, help = 'specify the end date')
    parser.add_argument('-n', '--new', action = 'store_true', help = 'retrieve new data')
    parser.add_argument('-o', '--open', action = 'store_true', help = 'open the file with default application (macOS only)')
    parser.add_argument('-p', '--pdf', action = 'store_true', help = 'generate PDF')
    parser.add_argument('-q', '--quiet', action = 'store_true', help = 'quiet mode')
    parser.add_argument('-s', '--small', action = 'store_true', help = 'make small size figures')
    parser.add_argument('-v', '--verbose', default = 0, action = 'count', help = 'increases verbosity level')
    parser.add_argument('-x', '--extra-large', action = 'store_true', help = 'set the figsize to be extra-large (2560 x 1440)')
    parser.add_argument('--dpi', default = 144, help = 'specify the DPI resoultion')
    parser.add_argument('--figsize', default = '(11.11112, 6.25)', help = 'specify the figsize')
    parser.add_argument('--format', default = 'png', help = 'specify the image format')
    args = parser.parse_args()
    # Post process the arguments
    if args.quiet:
        args.verbose = 0
    if args.symbols != '^OLD':
        args.symbols = [x.upper() for x in args.symbols]
    if args.pdf:
        args.format = 'pdf'
    if args.small:
        args.figsize = '(10.6667, 6)'
        args.dpi = 120
        args.days = 130
    if args.extra_large:
        args.figsize = '(10, 5.625)'
        args.dpi = 256
    # Internal variables
    figsize = literal_eval(args.figsize)
    # Show a summary if verbose
    if args.verbose:
        print('symbols = {}'.format(args.symbols))
        print('days = {}   end = {}   new = {}'.format(args.days, args.end, args.new))
    # Spin up the workers
    try:
        genfigs(args.symbols,
                days = args.days, end = args.end,
                verbose = args.verbose, image_format = args.format, dpi = args.dpi,
                color_scheme = args.color_scheme, open_preview = args.open,
                force_net = args.new, figsize = figsize)
    except KeyboardInterrupt:
        print('Exiting ...')
