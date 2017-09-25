import os
import datetime
import matplotlib
import chart
import data
import time
import mystyle

def refresh(days = 90, color_scheme = 'default', fig_folder = 'figs'):
    """
        Refresh all the png figures in fig_folder
    """
    # Stock data
    stock = data.get_old_data()

    # Check for output folder
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # Prepare the figure canvas
    print('Preparing figure ...')
    view = chart.Chart(days)
    view.set_xdata(stock.major_axis[-days:])

    # Go through the symbols in stock.minor_axis:
    for symbol in stock.minor_axis:
        # Update data
        view.set_data(stock[:, :, [symbol]])
        # Derive the filename
        filename = fig_folder + '/' + symbol + '.png'
        o = stock[['Open'], :, [symbol]].iloc[0, -1, 0]
        c = stock[['Close'], :, [symbol]].iloc[0, -1, 0]
        if c > o: 
            print('\033[38;5;46m{}\033[0m -> {}'.format(symbol.rjust(5), filename))
        else:
            print('\033[38;5;196m{}\033[0m -> {}'.format(symbol.rjust(5), filename))
        # Save an image
        view.savefig(filename)

#
#  M A I N
#
if __name__ == '__main__':
    refresh()
