import os
import datetime
import matplotlib
import chart
import data
import time
import mystyle

# Some global variables
K = 90                       # Number of days to show
figFolder = 'figs'            # Default folder to save figures

# Stock data
stock = data.get_old_data()

print('Preparing figure ...')
view = chart.Chart(K, color_scheme = 'night')
# view = chart.Chart(K)
view.set_xdata(stock.major_axis[-K:])

# Go through the symbols
# for symbol in stock.minor_axis:
for symbol in ['AABA']:
    # Update data
    view.set_data(stock[:, :, [symbol]])
    # Derive the filename
    filename = figFolder + '/' + symbol + '.png'
    o = stock[['Open'], :, [symbol]].iloc[0, -1, 0]
    c = stock[['Close'], :, [symbol]].iloc[0, -1, 0]
    if c > o: 
        print('\033[38;5;46m{}\033[0m -> {}'.format(symbol.rjust(5), filename))
    else:
        print('\033[38;5;196m{}\033[0m -> {}'.format(symbol.rjust(5), filename))
    # Save an image
    view.savefig(filename)
