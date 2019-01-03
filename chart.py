import matplotlib
import matplotlib.pyplot
import colorscheme
import stock
import numpy as np

# Minimum matplotlib version 2.0
if matplotlib.__version__ < '2.0':
    print('WARNING: Unexpected results may occur')

DEFAULT_SMA_SIZES = [10, 50, 200]
DEFAULT_RSI_PERIOD = 14
BACK_RECT = [0.075, 0.11, 0.83, 0.82]
MAIN_RECT = [0.075, 0.11, 0.83, 0.63]
RSI_RECT = [0.075, 0.74, 0.83, 0.19]
RSI_OB = 70
RSI_OS = 30

# The old way
def showChart(dat, sma_sizes = DEFAULT_SMA_SIZES, rsi_period = DEFAULT_RSI_PERIOD,
              use_adj_close = False, use_ema = False, skip_weekends = True, color_scheme = 'sunrise'):
    """
        showChart(dat, sma_size = [10, 50, 200], skip_weekends = True)
        - dat - Data frame from pandas-datareader
        - sma_sizes - Window sizes for SMA (sliding moving average)
        - skip_weekends - Skip plotting weekends and days with no data
    """
    linewidth = 1.0
    offset = 0.4

    fig = matplotlib.pyplot.figure()
    fig.patch.set_alpha(0.0)

    colormap = colorscheme.colorscheme(color_scheme)

    # Get the string description of open, high, low, close, volume
    desc = [x.lower() for x in dat.columns.get_level_values(0).tolist()]
    o_index = desc.index('open')
    h_index = desc.index('high')
    l_index = desc.index('low')
    c_index = desc.index('close')
    v_index = desc.index('volume')

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

    # Sort the data  (colums 1 ... 6) so that it is newest first
    # if quotes[1, 1] > quotes[1, 0]:
    #     # print('Resorting ... {}  {}'.format(quotes.shape, sma_sizes))
    #     quotes[:, 1:8] = quotes[::-1, 1:8]

    # Initialize an empty dictionary from keys based on sma size
    sma = dict.fromkeys(sma_sizes)
    n = max(sma_sizes)

    # Compute the SMA curves
    N = len(dat) - n - 1
    # print('N = {} - {} - 1 = {}'.format(len(dat), n, N))
    for k in sma.keys():
        sma[k] = stock.sma(quotes[:, 5], period = k)

    # Find the span of colums 2 to 5 (OHLC)
    nums = np.array(quotes[:N, 2:6]).flatten()
    ylim = [np.nanmin(nums), np.nanmax(nums)]

    dpi = 144.0

    rect = [(round(x * dpi) + 0.5) / dpi for x in BACK_RECT]
    axb = fig.add_axes(rect, frameon = False)
    axb.yaxis.set_visible(False)
    axb.xaxis.set_visible(False)

    # Main axis and volume axis
    rect = [(round(x * dpi) + 0.5) / dpi for x in MAIN_RECT]
    ax = fig.add_axes(rect, label = 'Quotes')
    ax.patch.set_visible(False)

    # Volume axis
    axv = fig.add_axes(rect, frameon = False)
    axv.patch.set_visible(False)
    axv.xaxis.set_visible(False)

    # RSI axis
    rect = [(round(x * dpi) + 0.5) / dpi for x in RSI_RECT]
    axr = matplotlib.pyplot.axes(rect, facecolor = None)
    axr.patch.set_visible(False)

    # SMA lines
    lines = []
    if use_ema:
        ma_label = 'EMA'
    else:
        ma_label = 'SMA'
    for k in sma.keys():
        if skip_weekends:
            # Plot the lines in indices; will replace the tics with custom label later
            sma_line = matplotlib.lines.Line2D(quotes[:N, 0], sma[k][:N], label = ma_label + ' ' + str(k))
        else:
            sma_line = matplotlib.lines.Line2D(quotes[:N, 1], sma[k][:N], label = ma_label + ' ' + str(k))
        lines.append(sma_line)
        ax.add_line(sma_line)
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
    rsi = stock.rsi(x, rsi_period)
    rsi = rsi[:-N-1:-1]

    color = colormap.line[3]
    rsi_line_25 = matplotlib.lines.Line2D([-1, N], [RSI_OS, RSI_OS], color = color, linewidth = 0.5, alpha = 0.33)
    rsi_line_50 = matplotlib.lines.Line2D([-1, N], [50.0, 50.0], color = color, linewidth = 1.0, alpha = 0.75, linestyle = '-.')
    rsi_line_75 = matplotlib.lines.Line2D([-1, N], [RSI_OB, RSI_OB], color = color, linewidth = 0.5, alpha = 0.33)
    axr.add_line(rsi_line_25)
    axr.add_line(rsi_line_50)
    axr.add_line(rsi_line_75)

    if skip_weekends:
        rsi_line = matplotlib.lines.Line2D(quotes[:N, 0], rsi, label = 'RSI {}'.format(rsi_period), color = color)
    else:
        rsi_line = matplotlib.lines.Line2D(quotes[:N, 1], rsi, label = 'RSI {}'.format(rsi_period), color = color)
    axr.add_line(rsi_line)

    axr.fill_between(range(N), rsi, RSI_OS, where = rsi <= RSI_OS, interpolate = True, color = color, alpha = 0.33, zorder = 3)
    axr.fill_between(range(N), rsi, RSI_OB, where = rsi >= RSI_OB, interpolate = True, color = color, alpha = 0.33, zorder = 3)

    # Round toward nice numbers
    if ylim[1] < 10:
        ylim[0] = np.floor(ylim[0])
        ylim[1] = np.ceil(ylim[1] * 2.0 + 1.0) * 0.5
    else:
        ylim[0] = np.floor(ylim[0] * 0.2) * 5.0
        ylim[1] = np.ceil(ylim[1] * 0.2) * 5.0

    majors = []
    vlines = []
    olines = []
    clines = []
    vrects = []

    # Add a dummy line at integer to get around tick locations not properly selected
    vline = matplotlib.lines.Line2D(xdata = (-1, -1), ydata = (0, 0), color = 'k', linewidth = linewidth)
    ax.add_line(vline)

    for q in quotes[:N]:
        if skip_weekends:
            i, t, o, h, l, c, v = q[:7]
            # Gather the indices of weeday == Monday
            if matplotlib.dates.num2date(t).weekday() == 0:
               majors.append(i)
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
        ax.add_line(vline)
        ax.add_line(oline)
        ax.add_line(cline)
        axv.add_patch(vrect)

    ax.set_xlim([N + 0.5, -1.5])
    axv.set_xlim([N + 0.5, -1.5])
    axr.set_xlim([N + 0.5, -1.5])

    axr.set_ylim([-1, 100])

    dr = RSI_OB - 50.0
    axr.set_yticks([RSI_OS - dr, RSI_OS, 50, RSI_OB, RSI_OB + dr])

    L = len(quotes)

    warned_x_tick = [3]

    def format_date(x, pos = None):
        if warned_x_tick[0] > 0 and abs(x - round(x)) > 1.0e-3:
            warned_x_tick[0] -= 1
            print('\033[38;5;220mWARNING: x = {} ticks are too far away from day boundaries.\033[0m'.format(x))
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
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))
        ax.xaxis.set_minor_locator(matplotlib.ticker.IndexLocator(1, 0))
        ax.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(5, majors[0] % 5 - 4))  # Use the latest Monday
        axr.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(5, majors[0] % 5 - 4))  # Use the latest Monday
    else:
        mondays = matplotlib.dates.WeekdayLocator(matplotlib.dates.MONDAY)      # major ticks on the mondays
        alldays = matplotlib.dates.DayLocator()                                 # minor ticks on the days
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
        axr.xaxis.set_major_locator(mondays)

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
    axb.imshow(np.linspace(0, 1, 100).reshape(-1, 1), cmap = cmap, extent = (-1, 1, -1, 1), aspect = 'auto')

    matplotlib.pyplot.setp(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

    lines[0].set_color(colormap.line[0])
    lines[1].set_color(colormap.line[1])
    lines[2].set_color(colormap.line[2])

    leg = ax.legend(handles = lines, loc = 'upper left', ncol = 3, frameon = False, fontsize = 9)
    for text in leg.get_texts():
        text.set_color(colormap.text)
    leg_rsi = axr.legend(handles = [rsi_line], loc = 'upper left', frameon = False, fontsize = 9)
    for text in leg_rsi.get_texts():
        text.set_color(colormap.text)

    ax.grid(alpha = colormap.grid_alpha, color = colormap.grid, linestyle = ':')
    axr.grid(alpha = colormap.grid_alpha, color = colormap.grid, linestyle = ':')
    ax.tick_params(axis = 'x', which = 'both', colors = colormap.text)
    ax.tick_params(axis = 'y', which = 'both', colors = colormap.text)
    axv.tick_params(axis = 'x', which = 'both', colors = colormap.text)
    axv.tick_params(axis = 'y', which = 'both', colors = colormap.text)
    axr.tick_params(axis = 'x', which = 'both', colors = colormap.text)
    axr.tick_params(axis = 'y', which = 'both', colors = colormap.text)
    ax.set_ylim(ylim)
    ax.yaxis.tick_right()
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_color(colormap.text)
        axv.spines[side].set_color(colormap.text)
        axr.spines[side].set_color(colormap.text)
    ax.spines['top'].set_visible(False)
    axv.spines['top'].set_visible(False)
    axr.spines['bottom'].set_visible(False)

    axr.set_title(dat.columns[0][1], color = colormap.text, weight = 'bold')

    for tic in axr.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False

    dic = {'figure':fig, 'axes':ax, 'lines':lines, 'volume_axis':axv, 'rsi_axis':axr, 'close':matplotlib.pyplot.close}
    return dic

# The new way
class Chart:
    """
        A chart class
    """
    def __init__(self, n, data = None, sma_sizes = DEFAULT_SMA_SIZES, rsi_period = DEFAULT_RSI_PERIOD,
                 use_adj_close = False, use_ema = True, skip_weekends = True, forecast = 0, color_scheme = 'sunrise'):
        linewidth = 1.0
        offset = 0.4

        self.n = n
        self.sma = dict.fromkeys(sma_sizes)
        self.symbol = ''
        self.colormap = colorscheme.colorscheme(color_scheme)
        self.skip_weekends = skip_weekends
        self.forecast = forecast
        self.rsi_period = rsi_period
        self.use_adj_close = use_adj_close
        self.use_ema = use_ema

        self.fig = matplotlib.pyplot.figure()
        self.fig.patch.set_alpha(0.0)

        dpi = 144.0

        rect = [(round(x * dpi) + 0.5) / dpi for x in BACK_RECT]
        self.axb = self.fig.add_axes(rect, frameon = False)
        self.axb.yaxis.set_visible(False)
        self.axb.xaxis.set_visible(False)
        
        rect = [(round(x * dpi) + 0.5) / dpi for x in MAIN_RECT]
        self.axq = self.fig.add_axes(rect, label = 'Quotes')
        self.axq.patch.set_visible(False)
        self.axq.yaxis.tick_right()
        
        self.axv = self.fig.add_axes(rect, frameon = False, sharex = self.axq)
        self.axv.patch.set_visible(False)
        self.axv.xaxis.set_visible(False)
        
        rect = [(round(x * dpi) + 0.5) / dpi for x in RSI_RECT]
        self.axr = self.fig.add_axes(rect, label = 'RSI')
        self.axr.patch.set_visible(False)

        # Backdrop gradient
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('backdrop', self.colormap.backdrop)
        self.im = self.axb.imshow(np.linspace(0, 1, 100).reshape(-1, 1), cmap = cmap, extent = (-1, 1, -1, 1), aspect = 'auto')
        self.st = self.axb.text(0, 0, self.symbol,
                                fontproperties = matplotlib.font_manager.FontProperties(style = 'normal', size = 100, weight = 'bold'),
                                color = self.colormap.background_text_color, alpha = self.colormap.background_text_alpha,
                                horizontalalignment = 'center', verticalalignment = 'center')

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
        self.rsi_line_50 = matplotlib.lines.Line2D([0, self.n + 1], [50.0, 50.0], color = color, linewidth = linewidth, alpha = 0.67, linestyle = '-.')
        self.rsi_line_75 = matplotlib.lines.Line2D([0, self.n + 1], [RSI_OB, RSI_OB], color = color, linewidth = 0.5 * linewidth, alpha = 0.5)
        self.axr.add_line(self.rsi_line_25)
        self.axr.add_line(self.rsi_line_50)
        self.axr.add_line(self.rsi_line_75)

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
                linewidth = linewidth,
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

        # Grid
        self.axq.grid(alpha = self.colormap.grid_alpha, color = self.colormap.grid, linestyle=':')
        self.axr.grid(alpha = self.colormap.grid_alpha, color = self.colormap.grid, linestyle=':')
        for side in ['top', 'bottom', 'left', 'right']:
            self.axq.spines[side].set_color(self.colormap.text)
            self.axv.spines[side].set_color(self.colormap.text)
            self.axr.spines[side].set_color(self.colormap.text)
        self.axq.spines['top'].set_visible(False)
        self.axv.spines['top'].set_visible(False)
        self.axr.spines['bottom'].set_visible(False)
        self.axq.tick_params(axis = 'x', which = 'both', colors = self.colormap.text)
        self.axq.tick_params(axis = 'y', which = 'both', colors = self.colormap.text)
        self.axv.tick_params(axis = 'x', which = 'both', colors = self.colormap.text)
        self.axv.tick_params(axis = 'y', which = 'both', colors = self.colormap.text)
        self.axr.tick_params(axis = 'x', which = 'both', colors = self.colormap.text)
        self.axr.tick_params(axis = 'y', which = 'both', colors = self.colormap.text)

        # Set the search limit here for x-tick lookup in matplotlib
        self.axq.xaxis.set_data_interval(-1.0, self.n + 2.0)
        self.axv.xaxis.set_data_interval(-1.0, self.n + 2.0)
        self.axr.xaxis.set_data_interval(-1.0, self.n + 2.0)
        dr = RSI_OB - 50.0
        self.axr.set_yticks([RSI_OS - dr, RSI_OS, 50, RSI_OB, RSI_OB + dr])
        self.axq.set_xlim([-1.5, self.n + 0.5])
        self.axv.set_xlim([-1.5, self.n + 0.5])
        self.axr.set_xlim([-1.5, self.n + 0.5])
        self.axq.set_ylim([0, 110])
        self.axv.set_ylim([0, 10])
        self.axr.set_ylim([0, 100])

        self.title = self.axr.set_title(self.symbol, color = self.colormap.text, weight = 'bold')

        if data is not None:
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
                print('\033[38;5;220mWARNING: x = {} ticks are too far away from day boundaries.\033[0m'.format(x))
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
            self.axq.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))
            self.axq.xaxis.set_minor_locator(matplotlib.ticker.IndexLocator(1, 0))
            # self.axq.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
            # self.axq.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors))
            self.axq.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(5, majors[-1] % 5 + 1))  # Use the last Monday
            self.axr.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(5, majors[-1] % 5 + 1))  # Use the last Monday
        else:
            mondays = matplotlib.dates.WeekdayLocator(matplotlib.dates.MONDAY)      # major ticks on the mondays
            alldays = matplotlib.dates.DayLocator()                                 # minor ticks on the days
            self.axq.xaxis.set_major_locator(mondays)
            self.axq.xaxis.set_minor_locator(alldays)
            self.axq.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
            self.axr.xaxis.set_major_locator(mondays)

        for tic in self.axr.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False

        matplotlib.pyplot.setp(self.axq.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

    def set_data(self, data):
        # Get the symbol from minor axis
        self.symbol = data.columns[0][1]                                         # Get it from column index ('open', 'AAPL')
        self.set_xdata(data.index)                                               # Populate the x-axis with datetime

        # Get the string description of open, high, low, close, volume
        desc = [x.lower() for x in data.columns.get_level_values(0).tolist()]
        o_index = desc.index('open')
        h_index = desc.index('high')
        l_index = desc.index('low')
        c_index = desc.index('close')
        v_index = desc.index('volume')

        # Build a flat array of OHLC and V data
        quotes = np.transpose([
            data.iloc[:, o_index].squeeze().tolist(),                            # 0 - Open
            data.iloc[:, h_index].squeeze().tolist(),                            # 1 - High
            data.iloc[:, l_index].squeeze().tolist(),                            # 2 - Low
            data.iloc[:, c_index].squeeze().tolist(),                            # 3 - Close
            np.multiply(data.iloc[:, v_index].squeeze(), 1.0e-6).tolist()        # 4 - Volume in millions
        ])

        if data.shape[0] < self.n:
            print('ERROR: Supplied data is less than the promised setup: {} vs {}'.format(self.n, data.shape[0]))
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
        self.rsi = stock.rsi(data.iloc[:, c_index].squeeze(), self.rsi_period)[-self.n:]
        color = self.colormap.line[3]
        self.rsi_line.set_ydata(self.rsi)
        self.rsi_fill_25.remove()
        self.rsi_fill_75.remove()
        self.rsi_fill_25 = self.axr.fill_between(range(self.n), self.rsi, RSI_OS, where = self.rsi <= RSI_OS, interpolate = True, color = color, alpha = 0.33, zorder = 3)
        self.rsi_fill_75 = self.axr.fill_between(range(self.n), self.rsi, RSI_OB, where = self.rsi >= RSI_OB, interpolate = True, color = color, alpha = 0.33, zorder = 3)

        # Legend position: upper right if SMA-N is increasing, upper left otherwise (Not in public API)
        sma = self.sma[list(self.sma)[-2]]
        if sma[-10] > sma[0]:
            self.leg_sma._loc = 2
            self.leg_rsi._loc = 2
        else:
            self.leg_sma._loc = 1
            self.leg_rsi._loc = 1

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

        # Update axis limits
        self.axq.set_ylim(qlim)
        self.axv.set_ylim(vlim)
        self.axv.set_yticks(ticks)
        self.axv.set_yticklabels(labels)
        self.title.set_text(self.symbol)
        self.st.set_text(self.symbol)

    def set_title(self, title):
        self.title.set_text(title)

    def close(self):
        matplotlib.pyplot.close(self.fig)

    def savefig(self, filename):
        self.fig.savefig(filename)
