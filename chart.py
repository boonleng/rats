import matplotlib
import matplotlib.pyplot
import numpy as np
import colorscheme
import stock

# Minimum matplotlib version 2.0
if matplotlib.__version__ < '2.0':
    print('WARNING: Unexpected results may occur')

DEFAULT_SMA_SIZES = [10, 50, 200]
DEFAULT_RSI_PERIOD = 14
DEFAULT_MACD_PERIODS = [9, 12, 26]
BACK_RECT = [0.075, 0.11, 0.83, 0.82]
RSI_RECT  = [0.075, 0.78, 0.83, 0.15]
MAIN_RECT = [0.075, 0.26, 0.83, 0.52]
MACD_RECT = [0.075, 0.11, 0.83, 0.15] 
RSI_OB = 70
RSI_OS = 30

# The typical way not not reusable
def showChart(dat, n = 130,
              sma_sizes = DEFAULT_SMA_SIZES,
              rsi_period = DEFAULT_RSI_PERIOD,
              macd_periods = DEFAULT_MACD_PERIODS,
              use_adj_close = False,
              skip_weekends = True,
              use_ema = True,
              figsize = (8.89, 5.0),
              color_scheme = 'sunrise'):
    """
        showChart(dat, sma_size = [10, 50, 200], skip_weekends = True)
        - dat - Data frame from pandas-datareader
        - sma_sizes - Window sizes for SMA (sliding moving average)
        - skip_weekends - Skip plotting weekends and days with no data
    """
    linewidth = 1.0
    offset = 0.4

    fig = matplotlib.pyplot.figure(figsize = figsize)
    fig.patch.set_alpha(0.0)

    colormap = colorscheme.colorscheme(color_scheme)

    # Get the string description of open, high, low, close, volume
    desc = [x.lower() for x in dat.columns.get_level_values(0).tolist()]
    o_index = desc.index('open')
    h_index = desc.index('high')
    l_index = desc.index('low')
    c_index = desc.index('close')
    v_index = desc.index('volume')

    # Sort the data  (colums 1 ... 6) so that it is newest first
    dat = dat.sort_index(axis = 0, ascending = False)

    # Build a flat array of OHLC and V data
    quotes = np.transpose([
        range(len(dat)),                                                     # index
        matplotlib.dates.date2num(dat.index),                                # datenum
        dat.iloc[:, o_index].squeeze().tolist(),                             # Open
        dat.iloc[:, h_index].squeeze().tolist(),                             # High
        dat.iloc[:, l_index].squeeze().tolist(),                             # Low
        dat.iloc[:, c_index].squeeze().tolist(),                             # Close
        np.multiply(dat.iloc[:, v_index].squeeze(), 1.0e-6).tolist()         # Volume in millions
    ])

    # Initialize an empty dictionary from keys based on sma size
    sma = dict.fromkeys(sma_sizes)
    m = max(sma_sizes)

    # Compute the SMA curves (data is in descending order, so we flip, compute, then flip again)
    for k in sma.keys():
        d = stock.sma(quotes[::-1, 5], period = k, length = len(dat))
        sma[k] = d[::-1]

    # Find the span of colums 2 to 5 (OHLC)
    N = min(max(len(dat), len(dat) - m - 1), n)
    #print('N = {} - {} - 1 = {} --> {}'.format(len(dat), m, len(dat) - m - 1, N))
    nums = np.array(quotes[:N, 2:6]).flatten()
    if all(np.isnan(nums)):
        ylim = [0.0, 1.0]
    else:
        ylim = [np.nanmin(nums), np.nanmax(nums)]

    axb = fig.add_axes(BACK_RECT, frameon = False)
    axb.yaxis.set_visible(False)
    axb.xaxis.set_visible(False)

    # Main axis for quotes and volume axis
    axq = fig.add_axes(MAIN_RECT, label = 'Quotes')
    axq.patch.set_visible(False)

    # Volume axis
    axv = fig.add_axes(MAIN_RECT, frameon = False)
    axv.patch.set_visible(False)
    axv.xaxis.set_visible(False)

    # RSI axis
    axr = matplotlib.pyplot.axes(RSI_RECT, facecolor = None)
    axr.patch.set_visible(False)

    # MACD axis
    axm = matplotlib.pyplot.axes(MACD_RECT, facecolor = None)
    axm.patch.set_visible(False)

    # SMA lines
    lines = []
    if use_ema:
        ma_label = 'EMA'
    else:
        ma_label = 'SMA'
    for k in sma.keys():
        if skip_weekends:
            # Plot the lines in indices; will replace the tics with custom label later
            sma_line = matplotlib.lines.Line2D(quotes[:N, 0], sma[k][:N], label = ma_label + ' ' + str(k), linewidth = linewidth)
        else:
            sma_line = matplotlib.lines.Line2D(quotes[:N, 1], sma[k][:N], label = ma_label + ' ' + str(k), linewidth = linewidth)
        lines.append(sma_line)
        axq.add_line(sma_line)
        if np.sum(np.isfinite(sma[k][:N])):
            y = np.nanpercentile(sma[k][:N], 25)
            if y < ylim[0]:
                ylim[0] = y
            y = np.nanpercentile(sma[k][:N], 75)
            if y > ylim[1]:
                ylim[1] = y
    if ylim[1] - ylim[0] < 10:
        ylim = [round(ylim[0]) - 0.5, round(ylim[1]) + 0.5]
    else:
        ylim = [round(ylim[0] * 0.2 - 1.0) * 5.0, round(ylim[1] * 0.2 + 1.0) * 5.0]

    # RSI line
    x = dat.iloc[:, c_index]
    if x.index[0] > x.index[1]:
        x = x[::-1]
    N_rsi = min(n, len(dat) - rsi_period)
    rsi = stock.rsi(x, rsi_period)
    rsi = rsi[:-N_rsi-1:-1]

    color = colormap.line[3]
    rsi_line_25 = matplotlib.lines.Line2D([-1, N], [RSI_OS, RSI_OS], color = color, linewidth = 0.5, alpha = 0.33)
    rsi_line_50 = matplotlib.lines.Line2D([-1, N], [  50.0,   50.0], color = color, linewidth = 1.0, alpha = 0.75, linestyle = '-.')
    rsi_line_75 = matplotlib.lines.Line2D([-1, N], [RSI_OB, RSI_OB], color = color, linewidth = 0.5, alpha = 0.33)
    axr.add_line(rsi_line_25)
    axr.add_line(rsi_line_50)
    axr.add_line(rsi_line_75)

    if skip_weekends:
        rsi_line = matplotlib.lines.Line2D(quotes[:N_rsi, 0], rsi, label = 'RSI {}'.format(rsi_period), color = color, linewidth = linewidth)
    else:
        rsi_line = matplotlib.lines.Line2D(quotes[:N_rsi, 1], rsi, label = 'RSI {}'.format(rsi_period), color = color, linewidth = linewidth)
    axr.add_line(rsi_line)

    axr.fill_between(range(N_rsi), rsi, RSI_OS, where = rsi <= RSI_OS, interpolate = True, color = color, alpha = 0.33, zorder = 3)
    axr.fill_between(range(N_rsi), rsi, RSI_OB, where = rsi >= RSI_OB, interpolate = True, color = color, alpha = 0.33, zorder = 3)

    # MACD lines
    macd, macd_ema, macd_div = stock.macd(x, macd_periods, length = n)
    macd = macd[::-1]
    macd_ema = macd_ema[::-1]
    macd_div = macd_div[::-1]
    if skip_weekends:
        macd_line = matplotlib.lines.Line2D(quotes[:n, 0], macd, label = 'MACD ({}, {}, {})'.format(macd_periods[0], macd_periods[1], macd_periods[2]), color = colormap.line[4], linewidth = linewidth)
        macd_line2 = matplotlib.lines.Line2D(quotes[:n, 0], macd_ema, label = 'MACD EMA', color = colormap.line[5], linewidth = linewidth)
    else:
        macd_line = matplotlib.lines.Line2D(quotes[:n, 1], macd, label = 'MACD ({}, {}, {})'.format(macd_periods[0], macd_periods[1], macd_periods[2]), color = colormap.line[4], linewidth = linewidth)
        macd_line2 = matplotlib.lines.Line2D(quotes[:n, 1], macd_ema, label = 'MACD EMA', color = colormap.line[5], linewidth = linewidth)
    axm.add_line(macd_line)
    axm.add_line(macd_line2)

    majors = []
    for i in range(N):
        if skip_weekends:
            k = quotes[i, 0]
            t = quotes[i, 1]
            # Gather the indices of weeday == Monday
            if matplotlib.dates.num2date(t).weekday() == 0:
               majors.append(k)
        else:
            k = quotes[i, 1]
        rect = matplotlib.patches.Rectangle(xy = (k - 0.5, 0.0),
            fill = True,
            snap = True,
            width = 1.0,
            height = macd_div[i],
            facecolor = colormap.bar,
            edgecolor = colormap.text,
            linewidth = linewidth,
            alpha = 0.35)
        axm.add_patch(rect)

    # Round toward nice numbers
    if ylim[1] < 10:
        ylim[0] = np.floor(ylim[0])
        ylim[1] = np.ceil(ylim[1] * 2.0 + 1.0) * 0.5
    else:
        ylim[0] = np.floor(ylim[0] * 0.2) * 5.0
        ylim[1] = np.ceil(ylim[1] * 0.2) * 5.0

    vlines = []
    olines = []
    clines = []
    vrects = []

    # Add a dummy line at integer to get around tick locations not properly selected
    axr.add_line(matplotlib.lines.Line2D(xdata = (-1, -1), ydata = (0, 0), color = 'k', linewidth = linewidth))
    axq.add_line(matplotlib.lines.Line2D(xdata = (-1, -1), ydata = (0, 0), color = 'k', linewidth = linewidth))
    axm.add_line(matplotlib.lines.Line2D(xdata = (-1, -1), ydata = (0, 0), color = 'k', linewidth = linewidth))

    for q in quotes[:N]:
        if skip_weekends:
            i, t, o, h, l, c, v = q[:7]
        else:
            k, i, o, h, l, c, v = q[:7]
        if c >= o:
            line_color = colormap.up
            bar_color = colormap.bar_up
        else:
            line_color = colormap.down
            bar_color = colormap.bar_down
        vline = matplotlib.lines.Line2D(xdata = (i, i), ydata = (l, h), color = line_color, linewidth = linewidth)
        oline = matplotlib.lines.Line2D(xdata = (i + offset, i), ydata = (o, o), color = line_color, linewidth = linewidth)
        cline = matplotlib.lines.Line2D(xdata = (i - offset, i), ydata = (c, c), color = line_color, linewidth = linewidth)
        vrect = matplotlib.patches.Rectangle(xy = (i - 0.5, 0.0),
            fill = True,
            width = 1.0,
            height = v,
            facecolor = bar_color,
            edgecolor = colormap.text,
            linewidth = 0.75,
            alpha = 0.33)
        vlines.append(vline)
        olines.append(oline)
        clines.append(cline)
        vrects.append(vrect)
        axq.add_line(vline)
        axq.add_line(oline)
        axq.add_line(cline)
        axv.add_patch(vrect)

    axq.set_xlim([N + 0.5, -1.5])
    axv.set_xlim([N + 0.5, -1.5])
    axr.set_xlim([N + 0.5, -1.5])
    axm.set_xlim([N + 0.5, -1.5])

    axr.set_ylim([-1, 100])

    dr = RSI_OB - 50.0
    axr.set_yticks([RSI_OS - dr, RSI_OS, 50, RSI_OB, RSI_OB + dr])

    # MACD range
    m = np.nanmax(np.abs(macd))
    mticks, mlim = ticks_lims_finder(m)
    axm.set_yticks(mticks)
    axm.set_ylim(mlim)

    L = len(quotes)

    warned_x_tick = [3]

    def format_date(x, pos = None):
        if warned_x_tick[0] > 0 and abs(x - round(x)) > 1.0e-3:
            warned_x_tick[0] -= 1
            print('\033[38;5;220mWARNING: x = {} ticks are too far away from day boundaries. Find the dummy line\033[0m'.format(x))
            if warned_x_tick[0] == 0:
                print('\033[38;5;220mWARNING message repeated 3 times.\033[0m'.format(x))
        index = int(x)
        # print('x = {}'.format(x))
        if x < 0:
            #print('Project to {} days from {}.'.format(-index, matplotlib.dates.num2date(quotes[0, 1]).strftime('%b %d')))
            k = 0
            t = quotes[0, 1]
            while (k < -index):
                t += 1.0
                weekday = matplotlib.dates.num2date(t).weekday()
                # Only count Mon through Friday
                if weekday >= 0 and weekday <= 4:
                    k += 1
                # print('index = {}   weekday {}   k = {}'.format(index, weekday, k))
            date = matplotlib.dates.num2date(t)
            #print('date -> {}'.format(date))
        elif index > L - 1:
            return ''
        else:
            date = matplotlib.dates.num2date(quotes[index, 1])
        # print('x = {}   index = {} --> {} ({})'.format(x, index, date.strftime('%b %d'), date.weekday()))
        return date.strftime('%b %d')

    if skip_weekends:
        axr.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(5, majors[0] % 5 - 4))  # Use the latest Monday
        axq.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(5, majors[0] % 5 - 4))  # Use the latest Monday
        axm.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(5, majors[0] % 5 - 4))  # Use the latest Monday
        axm.xaxis.set_minor_locator(matplotlib.ticker.IndexLocator(1, 0))
        axm.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))
    else:
        mondays = matplotlib.dates.WeekdayLocator(matplotlib.dates.MONDAY)      # major ticks on the mondays
        alldays = matplotlib.dates.DayLocator()                                 # minor ticks on the days
        axm.xaxis.set_major_locator(mondays)
        axm.xaxis.set_minor_locator(alldays)
        axm.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))

    # Volume bars to have the mean at around 10% of the vertical space
    v = np.nanmean(np.array(quotes[:, 6]))
    if v < 1.0:
        blim = [0, np.ceil(v * 10.0)]
    else:
        blim = [0, np.ceil(v) * 10.0]
    axv.set_ylim(blim)
    yticks = axv.get_yticks()
    new_ticks = []
    new_labels = []
    for ii in range(1, int(len(yticks) / 3 + 1)):
        new_ticks.append(yticks[ii])
        if yticks[1] < 1.0:
            new_labels.append(str(int(yticks[ii] * 100.0) * 10) + 'K')
        else:
            new_labels.append(str(int(yticks[ii])) + 'M')
    axv.set_yticks(new_ticks)
    axv.set_yticklabels(new_labels)
    axv.xaxis.set_visible(False)

    # Backdrop gradient
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('backdrop', colormap.backdrop)
    axb.imshow(np.linspace(0, 1, 100).reshape(-1, 1), cmap = cmap, extent = (0, 1, 0, 1), aspect = 'auto')
    axb.add_line(matplotlib.lines.Line2D([0.0, 1.0], np.multiply([1.0, 1.0], (RSI_RECT[1] - BACK_RECT[1]) / BACK_RECT[3]), color = '#000000', linewidth = 0.5, alpha = 0.5))
    axb.add_line(matplotlib.lines.Line2D([0.0, 1.0], np.multiply([1.0, 1.0], (MAIN_RECT[1] - BACK_RECT[1]) / BACK_RECT[3]), color = '#000000', linewidth = 0.5, alpha = 0.5))
    axb.set_ylim([0, 1])

    matplotlib.pyplot.setp(axm.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

    lines[0].set_color(colormap.line[0])
    lines[1].set_color(colormap.line[1])
    lines[2].set_color(colormap.line[2])

    leg = axq.legend(handles = lines, loc = 'upper left', ncol = 3, frameon = False, fontsize = 9)
    b = best_legend_loc(quotes[:N, 5][::-1], exclude = [[0, 0], [0, 1], [0, 2], [1, 1]])
    leg._loc = b
    for text in leg.get_texts():
        text.set_color(colormap.text)
    leg_rsi = axr.legend(handles = [rsi_line], loc = 'upper left', frameon = False, fontsize = 9)
    leg_rsi._loc = b
    for text in leg_rsi.get_texts():
        text.set_color(colormap.text)
    leg_macd = axm.legend(handles = [macd_line], loc = 'upper left', frameon = False, fontsize = 9)
    leg_macd._loc = best_legend_loc(macd_ema[::-1], exclude_middle = True)
    for text in leg_macd.get_texts():
        text.set_color(colormap.text)

    axq.grid(alpha = colormap.grid_alpha, color = colormap.grid, linestyle = ':')
    axr.grid(alpha = colormap.grid_alpha, color = colormap.grid, linestyle = ':')
    axm.grid(alpha = colormap.grid_alpha, color = colormap.grid, linestyle = ':')
    for ax in [axq, axv, axr, axm]:
        ax.tick_params(axis = 'x', which = 'both', colors = colormap.text)
        ax.tick_params(axis = 'y', which = 'both', colors = colormap.text)
    axq.set_ylim(ylim)
    axq.yaxis.tick_right()
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_color(colormap.text)
        axv.spines[side].set_color(colormap.text)
        axr.spines[side].set_color(colormap.text)
    axq.spines['top'].set_visible(False)
    axq.spines['bottom'].set_visible(False)
    axv.spines['top'].set_visible(False)
    axv.spines['bottom'].set_visible(False)
    axr.spines['bottom'].set_visible(False)
    axm.spines['top'].set_visible(False)

    axr.set_title(dat.columns[0][1], color = colormap.text, weight = 'bold')

    for tic in axr.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False

    for tic in axq.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False

    dic = {'figure':fig, 'axes':ax, 'lines':lines, 'volume_axis':axv, 'rsi_axis':axr, 'close':matplotlib.pyplot.close}
    return dic

# The new way
class Chart:
    """
        A chart class
    """
    def __init__(self, data = None, n = 130,
                 sma_sizes = DEFAULT_SMA_SIZES,
                 rsi_period = DEFAULT_RSI_PERIOD,
                 macd_periods = DEFAULT_MACD_PERIODS,
                 use_adj_close = False,
                 skip_weekends = True,
                 use_ema = True,
                 forecast = 0,
                 figsize = (8.89, 5.0),
                 dpi = 144,
                 color_scheme = 'sunrise'):
        linewidth = 1.25
        offset = 0.4

        if not data is None:
            self.n = min(len(data), n)
        else:
            self.n = n

        self.sma = dict.fromkeys(sma_sizes)
        self.symbol = ''
        self.colormap = colorscheme.colorscheme(color_scheme)
        self.skip_weekends = skip_weekends
        self.forecast = forecast
        self.rsi_period = rsi_period
        self.macd_periods = macd_periods
        self.use_adj_close = use_adj_close
        self.use_ema = use_ema

        self.fig = matplotlib.pyplot.figure(figsize = figsize, dpi = dpi)
        self.fig.patch.set_alpha(0.0)

        self.axb = self.fig.add_axes(BACK_RECT, frameon = False)
        self.axb.yaxis.set_visible(False)
        self.axb.xaxis.set_visible(False)
        
        self.axq = self.fig.add_axes(MAIN_RECT, label = 'Quotes')
        self.axq.patch.set_visible(False)
        self.axq.yaxis.tick_right()
        
        self.axv = self.fig.add_axes(MAIN_RECT, frameon = False, sharex = self.axq)
        self.axv.patch.set_visible(False)
        self.axv.xaxis.set_visible(False)
        
        self.axr = self.fig.add_axes(RSI_RECT, label = 'RSI')
        self.axr.patch.set_visible(False)

        self.axm = self.fig.add_axes(MACD_RECT, label = 'MACD')
        self.axm.patch.set_visible(False)

        # Backdrop gradient
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('backdrop', self.colormap.backdrop)
        self.im = self.axb.imshow(np.linspace(0, 1, 100).reshape(-1, 1), cmap = cmap, extent = (0, 1, 0, 1), aspect = 'auto')
        self.st = self.axb.text(0.5, 0.5, self.symbol,
                                fontproperties = matplotlib.font_manager.FontProperties(style = 'normal', size = 100, weight = 'bold'),
                                color = self.colormap.background_text, alpha = self.colormap.background_text_alpha,
                                horizontalalignment = 'center', verticalalignment = 'center')
        self.brs = [self.axb.add_line(matplotlib.lines.Line2D([0.0, 1.0], np.multiply([1.0, 1.0], (RSI_RECT[1] - BACK_RECT[1]) / BACK_RECT[3]),
                                     color = '#000000', linewidth = 0.5, alpha = 0.5))]
        self.brs.append(self.axb.add_line(matplotlib.lines.Line2D([0.0, 1.0], np.multiply([1.0, 1.0], (MAIN_RECT[1] - BACK_RECT[1]) / BACK_RECT[3]),
                                     color = '#000000', linewidth = 0.5, alpha = 0.5)))
        self.axb.set_ylim([0, 1])

        # SMA lines
        self.sma_lines = []
        if self.use_ema:
            ma_label = 'EMA'
        else:
            ma_label = 'SMA'
        for j, k in enumerate(self.sma.keys()):
            sma_line = matplotlib.lines.Line2D(range(self.n), np.multiply(range(self.n), k / self.n),
                                               label = ma_label + ' ' + str(k),
                                               color = self.colormap.line[j], linewidth = linewidth)
            self.axq.add_line(sma_line)
            self.sma_lines.append(sma_line)

        # RSI line
        color = self.colormap.line[3]
        y = np.multiply(range(self.n), 100.0 / self.n)
        self.rsi_line = matplotlib.lines.Line2D(range(self.n), y, label = 'RSI {}'.format(self.rsi_period), color = color, linewidth = linewidth)
        self.rsi_fill_25 = self.axr.fill_between(range(self.n), y, RSI_OS, where = y <= RSI_OS, facecolor = color, interpolate = True, alpha = 0.33)
        self.rsi_fill_75 = self.axr.fill_between(range(self.n), y, RSI_OB, where = y >= RSI_OB, facecolor = color, interpolate = True, alpha = 0.33)
        self.axr.add_line(self.rsi_line)
        self.rsi_line_25 = matplotlib.lines.Line2D([0, self.n + 1], [RSI_OS, RSI_OS], color = color, linewidth = 0.5 * linewidth, alpha = 0.5)
        self.rsi_line_50 = matplotlib.lines.Line2D([0, self.n + 1], [  50.0,   50.0], color = color, linewidth = 1.0 * linewidth, alpha = 0.7, linestyle = '-.')
        self.rsi_line_75 = matplotlib.lines.Line2D([0, self.n + 1], [RSI_OB, RSI_OB], color = color, linewidth = 0.5 * linewidth, alpha = 0.5)
        self.axr.add_line(self.rsi_line_25)
        self.axr.add_line(self.rsi_line_50)
        self.axr.add_line(self.rsi_line_75)

        # MACD lines
        self.macd_lines = []
        color = self.colormap.line[4]
        y = np.multiply(range(self.n), 2.0 / self.n) - 1.0
        macd_line = matplotlib.lines.Line2D(range(self.n), y, label = 'MACD ({}, {}, {})'.format(self.macd_periods[0], self.macd_periods[1], self.macd_periods[2]), color = self.colormap.line[4], linewidth = linewidth)
        self.axm.add_line(macd_line)
        self.macd_lines.append(macd_line)
        y += 0.1
        macd_line = matplotlib.lines.Line2D(range(self.n), y, label = 'EMA MACD {}'.format(self.macd_periods[0]), color = self.colormap.line[5], linewidth = linewidth)
        self.axm.add_line(macd_line)
        self.macd_lines.append(macd_line)
        self.macd_rects = []
        for i in range(self.n):
            rect = matplotlib.patches.Rectangle(xy = (i - 0.5, 0.0),
                fill = True,
                snap = True,
                width = 1.0,
                height = 5.0,
                facecolor = self.colormap.bar,
                edgecolor = self.colormap.text,
                linewidth = 1.0,
                alpha = 0.35)
            self.macd_rects.append(rect)
            self.axm.add_patch(rect)

        # Candles and bars
        self.majors = []
        self.vlines = []
        self.olines = []
        self.clines = []
        self.vrects = []
        for i in range(self.n):
            # print('x = {}'.format(i))
            vline = matplotlib.lines.Line2D(xdata = (i, i), ydata = (i - 5.0, i + 5.0), color = 'k', linewidth = linewidth)
            oline = matplotlib.lines.Line2D(xdata = (i - offset, i), ydata = (i - 2.0, i - 2.0), color = 'k', linewidth = linewidth)
            cline = matplotlib.lines.Line2D(xdata = (i + offset, i), ydata = (i + 2.0, i + 2.0), color = 'k', linewidth = linewidth)
            vrect = matplotlib.patches.Rectangle(xy = (i - 0.5, 0.0),
                fill = True,
                snap = True,
                width = 1.0,
                height = 10.0,
                facecolor = '#0000ff',
                edgecolor = self.colormap.text,
                linewidth = 1.0,
                alpha = 0.35)
            self.vlines.append(vline)
            self.olines.append(oline)
            self.clines.append(cline)
            self.vrects.append(vrect)
            self.axq.add_line(vline)
            self.axq.add_line(oline)
            self.axq.add_line(cline)
            self.axv.add_patch(vrect)

        # A forecast point
        line = matplotlib.lines.Line2D(xdata = (self.n + 10.0, self.n + 10.0), ydata = (100.0, 100.0), color = 'r', linewidth = linewidth)
        self.axq.add_line(line)

        # Legend
        self.leg_sma = self.axq.legend(handles = self.sma_lines, loc = 'upper left', ncol = 3, frameon = False, fontsize = 9)
        for text in self.leg_sma.get_texts():
            text.set_color(self.colormap.text)
        self.leg_rsi = self.axr.legend(handles = [self.rsi_line], loc = 'upper left', frameon = False, fontsize = 9)
        for text in self.leg_rsi.get_texts():
            text.set_color(self.colormap.text)
        self.leg_macd = self.axm.legend(handles = [self.macd_lines[0]], loc = 'upper left', frameon = False, fontsize = 9)
        for text in self.leg_macd.get_texts():
            text.set_color(self.colormap.text)

        # Grid
        for ax in [self.axq, self.axr, self.axm]:
            ax.grid(alpha = self.colormap.grid_alpha, color = self.colormap.grid, linestyle=':')
        for side in ['top', 'bottom', 'left', 'right']:
            for ax in [self.axq, self.axv, self.axr, self.axm]:
                ax.spines[side].set_color(self.colormap.text)
        for ax in [self.axq, self.axv, self.axr, self.axm]:
            ax.tick_params(axis = 'x', which = 'both', colors = self.colormap.text)
            ax.tick_params(axis = 'y', which = 'both', colors = self.colormap.text)
            ax.xaxis.set_data_interval(-1.0, self.n + 2.0)
            ax.set_xlim([-1.5, self.n + 0.5])
        for ax in [self.axq, self.axv, self.axm]:
            ax.spines['top'].set_visible(False)
        for ax in [self.axr, self.axq, self.axv]:
            ax.spines['bottom'].set_visible(False)


        # Set the search limit here for x-tick lookup in matplotlib
        dr = RSI_OB - 50.0
        self.axr.set_yticks([RSI_OS - dr, RSI_OS, 50, RSI_OB, RSI_OB + dr])
        self.axm.set_yticks([-20.0, -10.0, 0.0, 10.0, 20.0])
        self.axr.set_ylim([0, 100])
        self.axm.set_ylim([-22.0, 22.0])

        self.title = self.axr.set_title(self.symbol, color = self.colormap.text, weight = 'bold')

        if not data is None:
            self.set_data(data)

    def set_xdata(self, xdata):
        dates = list(xdata[-self.n:])
        if len(dates) != self.n:
            print('xdata must be at least the same length as the chart setup (n = {}).'.format(self.n))
            return
        dnums = matplotlib.dates.date2num(dates)

        # Gather the major indices of weeday == Monday
        majors = []
        for i, t in enumerate(dates):
            if t.weekday() == 0:
               majors.append(i)
        #n = majors[-1]
        #print('DEBUG: {}, ..., {} -> \033[38;5;214m{}\033[0m ({})'.format(majors[0], n, dates[n].strftime('%Y-%m-%d'), dates[n].weekday()))

        self.warned_x_tick = 3

        def format_date(x, pos = None):
            if self.warned_x_tick > 0 and abs(x - round(x)) > 1.0e-3:
                self.warned_x_tick -= 1
                print('\033[38;5;220mWARNING: x = {} ticks are too far away from day boundaries. Find the dummy line\033[0m'.format(x))
                if self.warned_x_tick == 0:
                    print('\033[38;5;220mWARNING message repeated 3 times.\033[0m'.format(x))
            index = int(x)
            # print('x = {}'.format(x))
            if x < 0:
                #print('Project to {} days from {}.'.format(-index, matplotlib.dates.num2date(dates[0]).strftime('%b %d')))
                k = 0
                t = dnums[0]
                while (k <= -index):
                    t = t - 1.0
                    weekday = matplotlib.dates.num2date(t).weekday()
                    # Only count Mon through Friday
                    if weekday >= 0 and weekday <= 4:
                        k += 1
                    #print('index = {}   weekday {}   k = {}'.format(index, weekday, k))
                date = matplotlib.dates.num2date(t)
                #print('date -> {}'.format(date))
            elif index > self.n - 1:
                #print('Extrapolate {} days from {}.'.format(index - self.n, matplotlib.dates.num2date(dates[-1]).strftime('%b %d')))
                k = 0
                t = dnums[-1]
                while (k <= index - self.n):
                    t += 1.0
                    weekday = matplotlib.dates.num2date(t).weekday()
                    # Only count Mon through Friday
                    if weekday >= 0 and weekday <= 4:
                        k += 1
                    #print('index = {}   weekday {}   k = {}'.format(index, weekday, k))
                date = matplotlib.dates.num2date(t)
            else:
                date = dates[index]
            #print('x = {}   index = {} --> {} ({})'.format(x, index, date.strftime('%b %d'), date.weekday()))
            return date.strftime('%b %d')

        if self.skip_weekends:
            #self.axq.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))
            #self.axq.xaxis.set_minor_locator(matplotlib.ticker.IndexLocator(1, 0))
            #self.axq.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
            #self.axq.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors))
            self.axq.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(5, majors[-1] % 5 + 1))  # Use the last Monday
            self.axr.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(5, majors[-1] % 5 + 1))  # Use the last Monday
            self.axm.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))
            self.axm.xaxis.set_minor_locator(matplotlib.ticker.IndexLocator(1, 0))
            self.axm.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(5, majors[-1] % 5 + 1))  # Use the last Monday
        else:
            mondays = matplotlib.dates.WeekdayLocator(matplotlib.dates.MONDAY)      # major ticks on the mondays
            alldays = matplotlib.dates.DayLocator()                                 # minor ticks on the days
            self.axq.xaxis.set_major_locator(mondays)
            self.axr.xaxis.set_major_locator(mondays)
            self.axm.xaxis.set_major_locator(mondays)
            self.axm.xaxis.set_minor_locator(alldays)
            self.axm.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))

        for tic in self.axr.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False

        for tic in self.axq.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False

        matplotlib.pyplot.setp(self.axm.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

    def set_data(self, X):
        # Get the symbol from minor axis
        self.symbol = X.columns[0][1]                                            # Get it from column index ('open', 'AAPL')
        self.set_xdata(X.index)                                                  # Populate the x-axis with datetime

        # Get the string description of open, high, low, close, volume
        desc = [x.lower() for x in X.columns.get_level_values(0).tolist()]
        o_index = desc.index('open')
        h_index = desc.index('high')
        l_index = desc.index('low')
        c_index = desc.index('close')
        v_index = desc.index('volume')

        # Build a flat array of OHLC and V data
        quotes = np.transpose([
            X.iloc[:, o_index].squeeze(),          # 0 - Open
            X.iloc[:, h_index].squeeze(),          # 1 - High
            X.iloc[:, l_index].squeeze(),          # 2 - Low
            X.iloc[:, c_index].squeeze(),          # 3 - Close
            X.iloc[:, v_index].squeeze()           # 4 - Volume
        ])
        quotes[: , -1] *= 1.0e-6                    # Volume to counts of millions

        if X.shape[0] < self.n:
            print('ERROR: Supplied data is less than the promised setup: {} vs {}'.format(self.n, X.shape[0]))
            return

        # self.axq.draw_artist(self.axq.patch)
        # self.axv.draw_artist(self.axv.patch)

        for k, q in enumerate(quotes[-self.n:, :]):
            o, h, l, c, v = q[:5]
            if c >= o:
                line_color = self.colormap.up
                bar_color = self.colormap.bar_up
                #print(' Open:{0:.2f} Close:{1:.2f} \033[38;5;46mUp\033[0m'.format(o, c))
            else:
                line_color = self.colormap.down
                bar_color = self.colormap.bar_down
                #print(' Open:{0:.2f} Close:{1:.2f} \033[38;5;196mDown\033[0m'.format(o, c))
            self.vlines[k].set_ydata((l, h))
            self.vlines[k].set_color(line_color)
            self.olines[k].set_ydata((o, o))
            self.olines[k].set_color(line_color)
            self.clines[k].set_ydata((c, c))    
            self.clines[k].set_color(line_color)
            self.vrects[k].set_height(v)
            self.vrects[k].set_facecolor(bar_color)
        # Replace the colors of the last portion if self.forecast > 0
        if self.forecast > 0:
            k = self.n - self.forecast
            for q in quotes[-self.forecast:, :]:
                o, h, l, c, v = q[:5]
                line_color = '#2b8aff'
                bar_color = '#cccc00'
                self.vlines[k].set_ydata((l, h))
                self.vlines[k].set_color(line_color)
                self.olines[k].set_ydata((o, o))           
                self.olines[k].set_color(line_color)
                self.clines[k].set_ydata((c, c))    
                self.clines[k].set_color(line_color)
                self.vrects[k].set_facecolor(bar_color)
                k = k + 1
        # self.axq.draw_artist(self.vlines[k])
        # self.axq.draw_artist(self.olines[k])
        # self.axq.draw_artist(self.clines[k])
        # self.axq.draw_artist(self.vrects[k])
        # print('k = {}'.format(k))

        # Find the span of (OHLC)
        nums = np.array(quotes[-self.n:, 0:4]).flatten()
        qlim = [np.nanmin(nums), np.nanmax(nums)]

        # Compute SMA and update qlim
        for j, k in enumerate(self.sma.keys()):
            if self.use_ema:
                self.sma[k] = stock.ema(quotes[:, 3], period = k, length = self.n)
            else:
                self.sma[k] = stock.sma(quotes[:, 3], period = k, length = self.n)
            self.sma_lines[j].set_ydata(self.sma[k])
            # self.axq.draw_artist(self.sma_lines[j])
            if np.sum(np.isfinite(self.sma[k])):
                qlim[0] = min(qlim[0], np.nanpercentile(self.sma[k], 25))
                qlim[1] = max(qlim[1], np.nanpercentile(self.sma[k], 75))
        if qlim[1] - qlim[0] < 10:
            qlim = [round(qlim[0]) - 0.5, round(qlim[1]) + 0.5]
        else:
            qlim = [round(qlim[0] * 0.2 - 1.0) * 5.0, round(qlim[1] * 0.2 + 1.0) * 5.0]

        # Compute RSI
        self.rsi_fill_25.remove()
        self.rsi_fill_75.remove()
        color = self.colormap.line[3]
        self.rsi = stock.rsi(X.iloc[:, c_index].squeeze(), self.rsi_period, length = self.n)
        self.rsi_line.set_ydata(self.rsi)
        self.rsi_fill_25 = self.axr.fill_between(range(self.n), self.rsi, RSI_OS, where = self.rsi <= RSI_OS, interpolate = True, color = color, alpha = 0.33, zorder = 3)
        self.rsi_fill_75 = self.axr.fill_between(range(self.n), self.rsi, RSI_OB, where = self.rsi >= RSI_OB, interpolate = True, color = color, alpha = 0.33, zorder = 3)
    
        # Compute MACD
        self.macd, self.macd_ema, self.macd_div = stock.macd(X.iloc[:, c_index].squeeze(), self.macd_periods, length = self.n)
        self.macd_lines[0].set_ydata(self.macd)
        self.macd_lines[1].set_ydata(self.macd_ema)
        div = self.macd - self.macd_ema
        for k, m in enumerate(div):
            self.macd_rects[k].set_height(m)

        # Legend position: use the close values to determine the best quadrant
        b = best_legend_loc(quotes[-self.n:, 3], exclude = [[0, 0], [0, 1], [0, 2], [1, 1], [2, 1]])
        self.leg_sma._loc = b
        self.leg_rsi._loc = b
        self.leg_macd._loc = best_legend_loc(self.macd_ema[-self.n:], exclude_middle = True)

        # Volume bars to have the mean at around 10% of the vertical space
        v = np.nanmean(quotes[-self.n:, 4])
        ticks = []
        labels = []
        vlim = [0, np.ceil(v * 10.0)]
        if v < 1.0:
            steps = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
            step = steps[np.argmin(np.abs(steps - v))]
            t = 0.0
            while t < 3.0 * v:
                t = t + step
                ticks.append(t)
                labels.append(str(int(t * 100)) + 'K')
        elif v < 1000.0:
            steps = np.array([1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500])
            step = steps[np.argmin(np.abs(steps - v))]
            t = 0.0
            while t < 3.0 * v:
                t = t + step
                ticks.append(t)
                labels.append(str(int(t)) + 'M')
        else:
            steps = np.array([500, 1000, 2000, 2500, 5000])
            step = steps[np.argmin(np.abs(steps - v))]
            t = 0.0
            while t < 3.0 * v:
                t = t + step
                ticks.append(t)
                labels.append(str(t * 0.001) + 'B')
        # print('step = {}'.format(step))

        # MACD range
        m = np.nanmax(np.abs(self.macd))
        mticks, mlim = ticks_lims_finder(m)

        # Update axis limits
        self.axq.set_ylim(qlim)
        self.axv.set_ylim(vlim)
        self.axm.set_ylim(mlim)
        self.axv.set_yticks(ticks)
        self.axm.set_yticks(mticks)
        self.axv.set_yticklabels(labels)
        self.title.set_text(self.symbol)
        self.st.set_text(self.symbol)

    def set_title(self, title):
        self.title.set_text(title)

    def close(self):
        matplotlib.pyplot.close(self.fig)

    def savefig(self, filename):
        self.fig.savefig(filename)

def ticks_lims_finder(n, count = 2, allowance = 1.2):
    if n == 0:
        print('Error. Input n must be > 0')
        return [-0.5, 0.0, 0.5], [-1.0, 1.0]
    # Get the value to be within 10
    x = 9.0 * count * allowance
    s = 0.1
    p = 0
    if n > x:
        while n >= x:
            n /= 10.0
            p += 1
    elif n <= count:
        while n <= count:
            n *= 10.0
            p -= 1
    ss = [9.0, 8.0, 7.5, 7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.0]
    c1 = np.round(np.divide(0.8 * n, ss))
    c2 = np.round(np.divide(0.9 * n, ss))
    c3 = np.round(np.divide(n, ss))    
    if any(c1 == 2.0):
        i = np.argmin(np.abs(c1 - count))
        s = ss[i]
    elif any(c2 == 2.0):
        i = np.argmin(np.abs(c2 - count))
        s = ss[i]
    elif any(c3 == 2.0):
        i = np.argmin(np.abs(c3 - count))
        s = ss[i]
    else:
        print('Error. Unable to find a good match.')
    s *= np.power(10.0, p)
    n *= np.power(10.0, p)
    ticks = np.arange(-count * s, 1.1 * count * s, s)
    lims = [-1.0, 1.0]
    for x in [count + 0.1, count + 0.2, count + 0.5, count + 0.7]:
        if n <= allowance * x * s:
            lims = np.array([-x * s, x * s])
            break
    #print(ticks, lims)
    return ticks, lims

def best_legend_loc(v, exclude = [], exclude_middle = False):
    n = len(v)
    u = np.arange(n)
    mask = np.isfinite(v)
    u = u[mask]
    v = v[mask]
    xx = np.linspace(np.min(u), np.max(u), 4)
    yy = np.linspace(np.min(v), np.max(v), 4)
    grid, _, _ = np.histogram2d(u, v, bins=(xx, yy))
    grid = grid.T
    # Some locations are forbidden
    if exclude_middle:
        exclude.append([1, 0])
        exclude.append([1, 1])
        exclude.append([1, 2])
        exclude.append([0, 1])
        exclude.append([2, 1])
    for coord in exclude:
        grid[coord[0], coord[1]] = n
    loc = [grid[2, 2],  # 1 - upper right
           grid[2, 0],  # 2 - upper left
           grid[0, 0],  # 3 - lower left
           grid[0, 2],  # 4 - lower right
           n,           # 5 - right
           grid[1, 0],  # 6 - center left
           grid[1, 2],  # 7 - center right
           grid[0, 1],  # 8 - lower center
           grid[2, 1],  # 9 - upper center
           grid[1, 1]]  # 10 - center
    return np.argmin(loc) + 1
