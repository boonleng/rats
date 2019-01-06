#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 08:49:39 2019

@author: Boonleng Cheong
"""

import data
import stock

# Get a DataFrame of the symbol of interest
df = data.get('nvda')
desc = [x.lower() for x in df.columns.get_level_values(0).tolist()]
i = desc.index('close')
x = df.iloc[-180:, i].values

#import importlib
#stock = importlib.reload(stock)
ema12 = stock.ema2(x, period=12)
ema26 = stock.ema2(x, period=26)
ema_fast = stock.ema(x, period=12)
ema_slow = stock.ema(x, period=26)
macd, macd_ema, macd_div = stock.macd(x)

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Arial', 'Helvetica']
matplotlib.rcParams['font.sans-serif'] = ['System Font', 'Verdana', 'Arial']
matplotlib.rcParams['figure.figsize'] = (8.89, 7.5)   # Change the size of plots
if matplotlib.rcParams['figure.dpi'] < 108:
	matplotlib.rcParams['figure.dpi'] = 108

import matplotlib.pyplot as plt
fig = plt.figure()
ax1, ax2 = fig.subplots(2, 1, sharex=True)
ax1.plot(ema12, label='EMA 12')
ax1.plot(ema26, label='EMA 26')
ax1.plot(ema_fast, label='EMA fast')
ax1.plot(ema_slow, label='EMA slow')
ax1.plot(x, label='Data')
ax1.grid()
legend = ax1.legend(loc='lower left', fontsize='large', ncol=3)
ax2.plot(macd, label='MACD')
ax2.plot(macd_ema, label='EMA9_MACD')
ax2.grid()
ax2.fill_between(range(len(macd_div)), macd_div, 0.0, alpha=0.2, facecolor='#ff0077', edgecolor=None)
fig.savefig('blob/test-macd.png')
print('blob/test-macd.png generated.')
