import numpy as np
import pandas as pd

def sma2(data, period = 10, length = None, ascending = False):
    """
        Compute SMA (Simple Moving Average)
    """
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

def sma(data, period = 10, length = None):
    """
        Compute SMA (Simple Moving Average)
    """
    if not isinstance(data, pd.Series):                                # Convert to a panda Series if needed
        series = pd.Series(data)
    else:
        series = data
    sma = series.rolling(window = period, min_periods = period).mean().tolist()
    if length is not None:
        sma = sma[-length:]
        if len(sma) < length:
            sma = np.concatenate((np.full(length - len(sma), np.nan), sma))
    return sma

def ema(data, period = 10, length = None):
    """
        Compute EMA (Exponential Moving Average)
    """
    if not isinstance(data, pd.Series):                                # Convert to a panda Series if needed
        series = pd.Series(data)
    else:
        series = data
    sma = series.rolling(window = period, min_periods = period).mean()[:period]
    rest = series[period:]
    ema = pd.concat([sma, rest]).ewm(span = period, adjust = False).mean().tolist()
    if length is not None:
        ema = ema[-length:]
        if len(ema) < length:
            ema = np.concatenate((np.full(length - len(ema), np.nan), ema))
    return ema

def rsi(data, period = 14):
    """
        Compute RSI (Relative Strength Index)
    """
    if not isinstance(data, pd.Series):                                # Convert to a panda Series if needed
        series = pd.Series(data)
    else:
        series = data
    delta = series.diff().dropna()                                     # Drop the 1st element since it is NAN
    u, d = delta.copy(), delta.copy() * -1.0                           # Preparing for gains (up) and losses (down)
    u[delta < 0.0] = 0.0                                               # Zero out the non-gain for gains
    d[delta > 0.0] = 0.0                                               # Zero out the non-loss for losses
    u[period] = np.mean(u[:period])                                    # First value is sum of avg gains
    d[period] = np.mean(d[:period])                                    # First value is sum of avg losses
    u = u.drop(u.index[:period - 1])
    d = d.drop(d.index[:period - 1])
    # With com(center of mass) = (period - 1) ==> alpha = 1 / period
    # y[k] = (1 - 1 / period) * y[k - 1] + (1 / period) * x[k]
    rs = u.ewm(com = period - 1, adjust = False).mean() / d.ewm(com = period - 1, adjust = False).mean()
    rs = np.nan_to_num(rs)
    return 100.0 - 100.0 / (1.0 + rs)

