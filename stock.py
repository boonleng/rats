import numpy as np
import pandas as pd

def sma2(data, period = 10, length = None):
    """
        Compute SMA (Simple Moving Average)
    """
    w = np.ones((period, ))
    w /= w.sum()
    n = len(data)
    ma = np.convolve(data, w, mode = 'full')[:n]
    ma[:period] = np.nan
    if not length is None:
        ma = ma[-length:]
    return ma

def ema2(data, period = 10, length = None):
    """
        Compute EMA (Exponential Moving Average)
    """
    w = np.exp(np.linspace(-1.0, 0.0, period))
    w /= w.sum()
    n = len(data)
    ma = np.convolve(data, w, mode = 'full')[:n]
    ma[:period] = np.nan
    if not length is None:
        ma = ma[-length:]
    return ma

def sma(data, period = 10, length = None):
    """
        Compute SMA (Simple Moving Average)
    """
    if not isinstance(data, pd.Series):                                # Convert to a panda Series if needed
        series = pd.Series(data)
    else:
        series = data
    sma = series.rolling(window = period, min_periods = period).mean().values
    if not length is None:
        if len(sma) < length:
            sma = np.concatenate((np.full(length - len(sma), np.nan), sma))
        else:
            sma = sma[-length:]
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
    ema = pd.concat([sma, rest]).ewm(span = period, adjust = False).mean().values
    if not length is None:
        if len(ema) < length:
            ema = np.concatenate((np.full(length - len(ema), np.nan), ema))
        else:
            ema = ema[-length:]
    return ema

def rsi(data, period = 14, length = None):
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
    rs = np.concatenate((np.full(period, 0.0), rs))
    if not length is None:
        if len(rs) < length:
            rs = np.concatenate((np.full(length - len(rs), np.nan), rs))
        else:
            rs = rs[-length:]
    return 100.0 - 100.0 / (1.0 + rs)

def macd(data, period_fast = 12, period_slow = 26, length = None):
    """
        Compute the MACD (Moving Average Convergence Divergence)
    """
    series = pd.Series(data)
    ema_fast = ema(series, period = period_fast, length = length)
    ema_slow = ema(series, period = period_slow, length = length)
    #ema_fast = ema2(data, period = period_fast, length = length)
    #ema_slow = ema2(data, period = period_slow, length = length)
    macd = ema_fast - ema_slow
    return ema_fast, ema_slow, macd
