import re
import glob
import pandas
# import data
import numpy as np

# quotes = data.get_old_data()

symbols = []

symFolder = 'symbols'
for file in glob.glob(symFolder + '/*.pkl'):
	m = re.search(symFolder + '/(.+?).pkl', file)
	if m:
		symbol = m.group(1)
		symbols.append(symbol)

df = pandas.read_pickle(file)

dd = np.empty([df.shape[0], df.shape[1], len(symbols)])

i = 0
for file in glob.glob(symFolder + '/*.pkl'):
	m = re.search(symFolder + '/(.+?).pkl', file)
	if m:
		df = pandas.read_pickle(file)
		dd[:, :, i] = df.values
		i = i + 1

quotes2 = pandas.Panel(data = np.transpose(dd, (1, 0, 2)), items = df.keys().tolist(), minor_axis = symbols, major_axis = df.index.tolist())
