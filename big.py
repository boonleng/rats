import os
import argparse
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot
import pandas_datareader
import requests_cache
import chart
import tensorflow as tf

# Some global variables
N = 1250;                     # Look at stock prices for the last N days
figFolder = 'figs'            # Default folder to save figures
sma_sizes = [10, 50, 100]     # SMA window sizes

# Some default plotting attributes
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Arial']
matplotlib.rcParams['font.sans-serif'] = ['System Font', 'Verdana', 'Arial']
matplotlib.rcParams['figure.figsize'] = (8, 4)   # Change the size of plots
matplotlib.rcParams['figure.dpi'] = 108

# Fang - GOOG, NFLX, AMZN, FB
# Chip - MU, AMAT, MRVL, NVDA
symbols = [
    '^DJI', '^SPX',
    'AAPL', 'TSLA',
    'GOOG', 'BIDU', 'MSFT',
    'NVDA', 'AMAT', 'MRVL', 'MU', 'AMD', 'INTC',
    'AMZN', 'NFLX', 'EBAY', 'BABA', 'BKS',
    'FB', 'TWTR', 'YELP',
    'NDLS', 'CMG', 'MCD',
    'S', 'T', 'VZ', 'TMUS', 'QCOM',
    'SBUX',
    'STX', 'WDC', 'DVMT', 'SNDK', 'HTCH', 'TXN', 'ADI', 'MCHP',
    'SNE',
    'C', 'V', 'BAC', 'WFC', 'AMTD',
    'BP', 'XON', 'CVX', 'OGE', 'JASO',
    'F', 'GM', 'TM'
]

max(sma_sizes)

# End day is always today, then roll back the maximum SMA window, plus another week
# end = datetime.date.today()
end = datetime.date(2017, 9, 15)
start = end - datetime.timedelta(days = (N + max(sma_sizes) + 7) * 7 / 5)

print('Loading data since ' + str(start) + ' ...')

session = requests_cache.CachedSession(cache_name = 'cache-big', backend = 'sqlite', expire_after = datetime.timedelta(days = 30))
stock = pandas_datareader.DataReader(symbols, 'yahoo', start, end, session = session)

# Set up the chart
print('Preparing figure ...')
view = chart.Chart(100)
view.set_xdata(stock.iloc[:, :, 0].index[:100])

# Go through the symbols
for symbol in symbols:
    print('Generating {} ...'.format(symbol))
    view.set_data(stock[:, :, symbol].iloc[:, -200:])
    view.set_title(symbol)
    filename = figFolder + '/' + symbol.lower() + '.png'
    view.savefig(filename)

# Create the model
L = stock.shape[2]
# x = tf.placeholder(tf.float32, [None, L])
# w = tf.Variable(tf.zeross[L, L])
# b = tf.Variable(tf.zeross[L, L])
# y = tf.matmul(x, W) + b

# y_ = tf.placeholder(tf.float32, [None, L])

# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label = y_, logits = y))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()

# for i in range(L):
