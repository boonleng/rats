import data
import chart
import multiprocessing

quotes = data.file()

charts = []
for i in range(4):
	charts.append(chart.Chart(n = 90))

def savefig_worker(chart, i):
	print('Saving fig {} ...'.format(chart.symbol))
	chart.savefig('figs/_mp{}.png'.format(i))

proclist = []
symbols = quotes.columns.levels[1].tolist()
for i, c in enumerate(charts):
	stock = data.get_frame(quotes, symbols[i])
	c.set_data(stock)
	p = multiprocessing.Process(target = savefig_worker, args = (c, i))
	proclist.append(p)
	p.start()

for p in proclist:
	p.join()
