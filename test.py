import data
import chart
import mystyle

stock = data.get_old_data(symbols = ['NVDA'])

# Some global variables
K = 90                       # Number of days to show
sma_sizes = [10, 50, 200]     # SMA window sizes

K = K + max(sma_sizes)
s = stock[:, stock.axes[1][-K:], :]

view = chart.showChart(s)

view['figure'].savefig('figs/NVDA.png')
