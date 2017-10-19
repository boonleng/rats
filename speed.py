import data
import chart
import mystyle
import time

# Some global variables
sma_sizes = [10, 50, 200]     # SMA window sizes
K = 90                        # Number of days to show

# Data length to ensure proper SMA calculation
L = K + max(sma_sizes)

# Get offline data
quotes = data.get_from_files()

# Method 0 or 1: 0 - recreate the chart everytime; 1 - update only the data portion
delta = [0, 0]
for method in [0, 1]:

    print('\033[38;5;190mMethod {}\033[0m:'.format(method))

    # Set up the chart in method 1
    if method is 1:
        view = chart.Chart(K)
        view.set_xdata(quotes.major_axis[-K:])

    # Make sure we get enough data so that all SMA curves are valid
    sss = quotes[:, quotes.axes[1][-L:], :]

    t1 = time.time()

    # Go through the symbols
    for symbol in quotes.minor_axis:
        if method is 0:
            # Recreate the chart everytime in method 0
            view = chart.showChart(sss[:, :, [symbol]])
        else:
            view.set_data(sss[:, :, [symbol]])
        print('-> \033[38;5;46m{}\033[0m'.format(symbol.rjust(4)))
        # view.savefig('figs/_speed.png')
        # Destroy the chart everytime in method 0
        if method is 0:
            view['close'](view['figure'])

    t0 = time.time()

    delta[method] = t0 - t1

    print('Elapsed time: {0:.3f} s'.format(t0 - t1))

print('Method 0 : Method 1 = {0:.3f}'.format(delta[0] / delta[1]))
