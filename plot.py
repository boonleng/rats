import os
import multiprocessing
import argparse
import data
import chart
import mystyle
import time

def savefig(chart, data_frame, filename):
    chart.set_data(data_frame)
    chart.savefig(filename)

def genfigs(symbols, days = 90, sma_sizes = chart.DEFAULT_SMA_SIZES, folder = 'figs',
            color_scheme = 'default', image_format = 'png', verbose = 0,
            open_preview = False, offline = False):
    # Get the latest data
    if symbols == '^OLD':
        stock = data.get_from_files()
    elif offline is True:
        stock = data.get_from_files(symbols = symbols)
    else:
        # Total data length to retrieve to have complete valid SMA
        L = days + max(sma_sizes);
        if verbose:
            print('Retrieving data for {} for L = {} ...'.format(symbols, L))
        stock = data.get_from_net(symbols, days = int(L * 1.6))

    # Show parts of the data if we are in verbose mode
    if verbose > 1:
        print(stock)
        print(stock.iloc[:, ::-1, 0].head())

    # Get the core count
    batch_size = min(multiprocessing.cpu_count(), len(stock.minor_axis))
    
    # Set up the chart
    if verbose:
        print('Preparing background (batch size = {}) ...'.format(batch_size))
    views = []
    for _ in range(batch_size):
        view = chart.Chart(days, color_scheme = color_scheme)
        view.set_xdata(stock.major_axis)
        views.append(view)

    # Create the output folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Go through the symbols
    t1 = time.time()
    procs = []
    command = 'open'
    for i, symbol in enumerate(stock.minor_axis):
        data_frame = stock[:, :, [(symbol)]]
        if verbose > 1:
            print(data_frame)
        # Derive the filename
        filename = folder + '/' + symbol + '.' + image_format
        delta = data_frame[['Open', 'Close'], :, :].iloc[:, -1, 0].diff().tolist()[-1]
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
    
    print('Elapsed time: {0:.3f} s'.format(t0 - t1))

    if open_preview:
        os.system(command)


#
#  M A I N
#
if __name__ == '__main__':
    examples = '''
    python plot.py AAPL BIDU
    python plot.py NVDA TSLA -z -o'''
    parser = argparse.ArgumentParser(prog = 'plot',
                                     usage = examples)
    parser.add_argument('symbols', default = '^OLD', nargs = '+', help = 'specify symbols, e.g., NVDA TSLA AAPL')
    parser.add_argument('-v', '--verbose', default = 1, action = 'count', help = 'increases verbosity level')
    parser.add_argument('-c', '--color-scheme', default = 'default', help = 'specify color scheme to use (sunrise, sunset, night).')
    parser.add_argument('-o', '--open-preview', action = 'store_true', help = 'open Preview (macOS only)')
    parser.add_argument('-p', '--pdf', action = 'store_true', help = 'generate PDF.')
    parser.add_argument('-q', '--quiet', action = 'store_true', help = 'quiet mode.')
    parser.add_argument('-z', '--offline', action = 'store_true', help = 'use offline data only')
    args = parser.parse_args()

    if args.quiet:
        args.verbose = 0

    if args.symbols != '^OLD':
        args.symbols = [x.upper() for x in args.symbols]

    if args.pdf:
        args.format = 'pdf'
    else:
        args.format = 'png'

    if args.verbose > 1:
        print('symbols = {}'.format(args.symbols))

    try:
        genfigs(args.symbols,
                verbose = args.verbose, image_format = args.format,
                color_scheme = args.color_scheme, open_preview = args.open_preview,
                offline = args.offline)
    except KeyboardInterrupt:
        print('Exiting ...')
