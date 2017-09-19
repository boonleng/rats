import os
import datetime
import matplotlib
import chart
import data
import time

# Some global variables
K = 90                       # Number of days to show
figFolder = 'figs'            # Default folder to save figures
sma_sizes = [10, 50, 100]     # SMA window sizes

# Some default plotting attributes
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Arial']
matplotlib.rcParams['font.sans-serif'] = ['System Font', 'Verdana', 'Arial']
matplotlib.rcParams['figure.figsize'] = (7, 4)   # Change the size of plots
matplotlib.rcParams['figure.dpi'] = 108

stock = data.get()

print('Preparing figure ...')
view = chart.Chart(K)
view.set_xdata(stock.major_axis[-K:])

# Make sure we get enough data so that all SMA curves are valid
K = K + max(sma_sizes)
sss = stock[:, stock.axes[1][-K:], :]
print(sss)

# Go through the symbols
for symbol in stock.minor_axis:
    # Update data
    view.set_data(sss[:, :, [symbol]])
    # Derive the filename
    filename = figFolder + '/' + symbol + '.png'
    print('\033[38;5;46m{}\033[0m -> {}'.format(symbol.rjust(5), filename))
    # Save an image
    view.savefig(filename)
