import os
import argparse
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot
import pandas_datareader
import requests_cache
import chart

days = 1

end = datetime.date.today()
start = end - datetime.timedelta(days = days)
session = requests_cache.CachedSession(cache_name = 'cache', backend = 'sqlite', expire_after = datetime.timedelta(days = 1))
stock = pandas_datareader.DataReader(['TSLA', 'GOOG'], 'google', start, end, session = session)

if stock.major_axis[1] < stock.major_axis[0]:
    print('Need to resort')
