import os
import argparse
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot
import pandas_datareader
import requests_cache
import chart

days = 200

end = datetime.date.today()
# start = end - datetime.timedelta(days = days)
start = end - datetime.timedelta(days = int(1.6 * days))
# session = requests_cache.CachedSession(cache_name = 'cache', backend = 'sqlite', expire_after = datetime.timedelta(hours = 1))
stock = pandas_datareader.DataReader(['TSLA', 'GOOG'], 'yahoo', start, end)
# stock = pandas_datareader.DataReader(['.INX'], 'google', start, end, session = session)

# if stock.major_axis[1] < stock.major_axis[0]:
#     print('Need to resort')
#     stock = stock.sort_index(axis = 1, ascending = True)
