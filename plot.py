import os
import argparse
import data
import chart
import mystyle

def refresh():
    genfigs('^OLD')

def genfigs(symbols, days = 90, sma_sizes = [10, 50, 100], folder = 'figs',
            color_scheme = 'default', image_format = 'png', verbose = 0,
            open_preview = False):
    # Get the latest data
    if symbols == '^OLD':
        stock = data.get_old_data()
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

    # Set up the chart
    if verbose:
        print('Preparing figure ...')
    view = chart.Chart(days, color_scheme = color_scheme)
    view.set_xdata(stock.major_axis)

    # Create the output folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Go through the symbols
    command = 'open'
    for symbol in stock.minor_axis:
        if verbose > 1:
            print(stock[:, :, [(symbol)]])
        # Update data
        view.set_data(stock[:, :, [(symbol)]])
        # Derive the filename
        filename = folder + '/' + symbol + '.' + image_format
        o = stock[['Open'], :, [symbol]].iloc[0, -1, 0]
        c = stock[['Close'], :, [symbol]].iloc[0, -1, 0]
        if c > o:
            print('\033[38;5;46m{}\033[0m -> \033[38;5;206m{}\033[0m'.format(symbol.rjust(5), filename))
        else:
            print('\033[38;5;196m{}\033[0m -> \033[38;5;206m{}\033[0m'.format(symbol.rjust(5), filename))
        # Save an image
        view.savefig(filename)
        command = command + ' ' + filename

    if open_preview:
        os.system(command)


#
#  M A I N
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'main')
    parser.add_argument('-v', '--verbose', default = 1, action = 'count', help = 'increases verbosity level')
    parser.add_argument('-s', '--symbols', default = '^OLD', nargs = '+', help = 'specify symbols, e.g., -s NVDA TSLA AAPL')
    parser.add_argument('-c', '--color-scheme', default = 'default', help = 'specify color scheme to use.')
    parser.add_argument('-o', '--open-preview', action = 'store_true', help = 'open Preview')
    parser.add_argument('-p', '--pdf', action = 'store_true', help = 'generate PDF.')
    parser.add_argument('-q', '--quiet', action = 'store_true', help = 'quiet mode.')
    args = parser.parse_args()
    if args.quiet:
        args.verbose = 0
    if args.symbols != '^OLD':
        args.symbols = [x.upper() for x in args.symbols]
    if args.pdf:
        args.format = 'pdf'
    else:
        args.format = 'png'

    # print('symbols = {}'.format(args.symbols))
    try:
        genfigs(args.symbols,
                verbose = args.verbose, image_format = args.format,
                color_scheme = args.color_scheme, open_preview = args.open_preview)
    except KeyboardInterrupt:
        print('Exiting ...')
