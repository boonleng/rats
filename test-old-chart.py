import data
import chart

stock = data.get_from_files(symbols = ['NVDA'])

# Some global variables
K = 130                       # Number of days to show
sma_sizes = [10, 50, 200]     # SMA window sizes

L = K + max(sma_sizes)
s = stock.iloc[-L:]

view = chart.showChart(s)

view['figure'].savefig('blob/NVDA_test.png')
