import os
import argparse
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot
import pandas_datareader
import requests_cache
import chart
import joblib
import multiprocessing

# Some global variables
N = 100;                      # Look at stock prices for the last N days
figFolder = 'figs'            # Default folder to save figures
sma_sizes = [10, 50, 100]     # SMA window sizes

symbols = ["AAPL", "TSLA", "NVDA", "BIDU", "BABA"]

# Some default plotting attributes
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Arial']
matplotlib.rcParams['font.sans-serif'] = ['System Font', 'Verdana', 'Arial']
matplotlib.rcParams['figure.figsize'] = (7, 4)   # Change the size of plots
matplotlib.rcParams['figure.dpi'] = 108

if not os.path.exists(figFolder):
    os.makedirs(figFolder)

# End day is always today, then roll back the maximum SMA window, plus another week
end = datetime.date.today()
start = end - datetime.timedelta(days = (N + max(sma_sizes) + 7) * 7 / 5)
session = requests_cache.CachedSession(cache_name = 'cache', backend = 'sqlite', expire_after = datetime.timedelta(days = 1))
stock = pandas_datareader.DataReader(symbols, "yahoo", start, end, session = session)

def showChart(stock, sym):
    view = chart.showChart(stock[:, :, sym], sma_sizes = sma_sizes)
    view['title'] = view['axes'].set_title(sym)
    view['figure'].savefig(figFolder + '/' + sym.lower() + '.png')

def main(verbose = 0):
    max(sma_sizes)

    num_cores = multiprocessing.cpu_count()
    print('Number of cores = {}'.format(num_cores))


    if args.verbose:
        print(start)


    results = joblib.Parallel(n_jobs = num_cores/2)(joblib.delayed(showChart)(sym) for sym in symbols)

    # for sym in symbols:
    #     print('Generating chart for {}'.format(sym))
    #     showChart(stock, sym)

#
#  M A I N
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "main")
    parser.add_argument('-v', '--verbose', default = 0, action = 'count', help = 'increases verbosity level')
    args = parser.parse_args()

    # print('Version {}'.format(sys.version_info))

    try:
        main(verbose = args.verbose)
    except KeyboardInterrupt:
        print('Exiting ...')
