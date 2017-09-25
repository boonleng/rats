import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot
import mystyle
import chart
import data
import time
# import tensorflow as tf

# Some global variables
K = 100                       # Number of days to show
sma_sizes = [10, 50, 100]     # SMA window sizes

quotes = data.get_old_data()

# Method 0 or 1
for method in [0, 1]:

    print('Method {}:'.format(method))

    # Set up the chart
    if method is 1:
        print('Preparing figure ...')
        view = chart.Chart(K)
        view.set_xdata(quotes.major_axis[-K:])
        #matplotlib.pyplot.show(block = False)

    # Make sure we get enough data so that all SMA curves are valid
    K = K + max(sma_sizes)
    sss = quotes[:, quotes.axes[1][-K:], :]
    print(sss)

    # Go through the symbols
    t1 = time.time()
    for symbol in quotes.minor_axis:
        if method is 0:
            view = chart.showChart(sss[:, :, [symbol]], color_scheme = 'sunrise')
        else:
            view.set_data(sss[:, :, [symbol]])
            #view.fig.canvas.draw()
            #view.fig.canvas.flush_events()
            #matplotlib.pyplot.pause(0.001)

        print('-> \033[38;5;46m{}\033[0m'.format(symbol.rjust(4)))
        # Close it if needed
        if method is 0:
            matplotlib.pyplot.close(view['figure'])

    t0 = time.time()

    print('Elapsed time: {0:.3f} s'.format(t0 - t1))
