import numpy as np
import pandas as pd

def sma(data, period = 10, length = None, ascending = False):
    sma = np.convolve(data, np.ones((period, )) / period, mode = 'valid')
    if length is None:
        length = len(data)
    if len(sma) < length:
        if ascending:
            sma = np.concatenate((np.full(length - len(sma), np.nan), sma))
        else:
            sma = np.pad(sma, (0, period - 1), mode = 'constant', constant_values = np.nan)
    else:
        sma = sma[-length:]
    return sma

def rsi(series, period = 14):
    delta = series.diff().dropna()              # Drop the 1st since it is NAN
    u, d = delta.copy(), delta.copy() * -1.0
    u[delta < 0.0] = 0.0
    d[delta > 0.0] = 0.0
    u[period] = np.mean(u[:period])             # First value is sum of avg gains
    u = u.drop(u.index[:period - 1])
    d[period] = np.mean(d[:period])             # First value is sum of avg losses
    d = d.drop(d.index[:period - 1])
    rs = u.ewm(com = period - 1, adjust = False).mean() / d.ewm(com = period - 1, adjust = False).mean()
    rs = np.nan_to_num(rs)
    return 100.0 - 100.0 / (1.0 + rs)
