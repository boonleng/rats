import re
import glob
import pandas
import data

quotes = data.get_old_data()

symFolder = 'symbols'
for file in glob.glob(symFolder + '/*.pkl'):
	m = re.search(symFolder + '/(.+?).pkl', file)
	if m:
		symbol = m.group(1)
		print(file, symbol)
		df = pandas.read_pickle(file)
