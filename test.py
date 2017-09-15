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
figFolder = 'figs'            # Default folder to save figures
sma_sizes = [10, 50, 100]     # SMA window sizes

symbols = ["AAPL", "TSLA", "NVDA", "BIDU", "AMZN", "BABA", "NFLX", "MSFT"]

# Some default plotting attributes
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Arial']
matplotlib.rcParams['font.sans-serif'] = ['System Font', 'Verdana', 'Arial']
matplotlib.rcParams['figure.figsize'] = (7, 4)   # Change the size of plots
matplotlib.rcParams['figure.dpi'] = 108

if not os.path.exists(figFolder):
    os.makedirs(figFolder)


max(sma_sizes)

# End day is always today, then roll back the maximum SMA window, plus another week
end = datetime.date.today()
start = end - datetime.timedelta(days = (N + max(sma_sizes) + 7) * 7 / 5)

session = requests_cache.CachedSession(cache_name = 'cache', backend = 'sqlite', expire_after = datetime.timedelta(days = 3))
stock = pandas_datareader.DataReader(symbols, "yahoo", start, end, session = session)



sym = 'AAPL'
view = chart.showChart(stock[:, :, sym], sma_sizes = sma_sizes)
view['title'] = view['axes'].set_title(sym)
# view['figure'].savefig(figFolder + '/' + sym.lower() + '.png')
matplotlib.pyplot.show()
