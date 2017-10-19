import data
import chart
import mystyle

stock = data.get_from_files(symbols = ['NVDA'])

# Some global variables
K = 90                       # Number of days to show
sma_sizes = [10, 50, 200]     # SMA window sizes

L = K + max(sma_sizes)
s = stock[:, stock.axes[1][-L:], :]

view = chart.showChart(s)

view['figure'].savefig('figs/NVDA_test.png')
