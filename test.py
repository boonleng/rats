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

# Some default plotting attributes
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Arial']
matplotlib.rcParams['font.sans-serif'] = ['System Font', 'Verdana', 'Arial']
matplotlib.rcParams['figure.figsize'] = (8, 4)   # Change the size of plots
matplotlib.rcParams['figure.dpi'] = 108

if not os.path.exists(figFolder):
    os.makedirs(figFolder)

symbols = ['AAPL', 'TSLA', 'NVDA']

days = N + 100;
# if args.verbose:
#     print('Retrieving data for {} for {} days ...'.format(args.symbols, days))
# End day is always today, then roll back the maximum SMA window, plus another week
end = datetime.date.today()
start = end - datetime.timedelta(days = days)
session = requests_cache.CachedSession(cache_name = 'cache', backend = 'sqlite', expire_after = datetime.timedelta(days = 1))
stock = pandas_datareader.DataReader(symbols, 'google', start, end, session = session)
# Google reports at least 250 days, truncate to desired length
if stock.shape[1] > days:
    print('Truncating data from {} to {} ...'.format(stock.shape[1], days))
    stock = stock.iloc[:, -days:, :]


view = chart.Chart(100)
view.set_xdata(stock.iloc[:, :, 0].index[-100:])

for sym in symbols:
	view.set_data(stock[:, :, sym])
	filename = figFolder + '/' + sym.lower() + '.png'
	view.savefig(filename)
	#os.system('open ' + filename)

# for symbol in args.symbols:
#     if args.verbose:
#         print('Generating chart for {}'.format(symbol))
#     showChart(symbol, stock, color_scheme = args.color_scheme)
