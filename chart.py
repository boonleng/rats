import numpy as np
import matplotlib
import matplotlib.pyplot
import colorscheme

DEFAULT_SMA_SIZES = [10, 50, 200]
BACK_RECT = [0.075, 0.12, 0.83, 0.78]
MAIN_RECT = [0.075, 0.12, 0.83, 0.58]
RSI_RECT = [0.075, 0.70, 0.83, 0.20]

def RSI(series, period = 14):
    delta = series.diff().dropna()              # Drop the 1st since it is NAN
    u, d = delta.copy(), delta.copy() * -1.0
    u[delta < 0.0] = 0.0
    d[delta > 0.0] = 0.0
    u[period] = np.mean(u[:period])             # First value is sum of avg gains
    u = u.drop(u.index[:period - 1])
    d[period] = np.mean(d[:period])             # First value is sum of avg losses
    d = d.drop(d.index[:period - 1])
    rs = u.ewm(com = period - 1, adjust = False).mean() / d.ewm(com = period - 1, adjust = False).mean()
    return 100.0 - 100.0 / (1.0 + rs)

# The old way
def showChart(panel, sma_sizes = DEFAULT_SMA_SIZES, rsi_period = 14, skip_weekends = True, color_scheme = 'sunrise'):
    """
        showChart(dat, sma_size = [10, 50, 200], skip_weekends = True)
        - dat - Data frame from pandas-datareader
        - sma_sizes - Window sizes for SMA (sliding moving average)
        - skip_weekends - Skip plotting weekends and days with no data
    """
    fig = matplotlib.pyplot.figure()
    fig.patch.set_alpha(0.0)

    colormap = colorscheme.colorscheme(color_scheme)

    # Get the first frame
    dat = panel.iloc[:, :, 0]
    quotes = np.transpose([
        list(range(len(dat))),
        list(matplotlib.dates.date2num(dat.index.tolist())),
        dat.loc[:, 'Open'].tolist(),
        dat.loc[:, 'High'].tolist(),
        dat.loc[:, 'Low'].tolist(),
        dat.loc[:, 'Close'].tolist(),
        np.multiply(dat.loc[:, 'Volume'], 1.0e-6).tolist()
    ])

    # Sort the data  (colums 1 ... 6) so that it is newest first
    if quotes[1, 1] > quotes[1, 0]:
        # print('Resorting ... {}  {}'.format(quotes.shape, sma_sizes))
        quotes[:, 1:8] = quotes[::-1, 1:8]

    # Initialize an empty dictionary from keys based on sma size
    sma = dict.fromkeys(sma_sizes)
    n = 0
    for num in sma.keys():
        n = max(n, num)

    # Compute the SMA curves
    N = len(dat) - n - 1
    # print('N = {} - {} - 1 = {}'.format(len(dat), n, N))
    for k in sma.keys():
        sma[k] = np.convolve(quotes[:, 5], np.ones((k, )) / k, mode = 'valid')
        sma[k] = np.pad(sma[k], (0, k - 1), mode = 'constant', constant_values = np.nan)

    # Find the span of colums 2 to 5 (OHLC)
    nums = np.array(quotes[:N, 2:6]).flatten()
    ylim = [np.nanmin(nums), np.nanmax(nums)]

    # Main axis and volume axis
    rect = MAIN_RECT
    rect = [round(x * 72.0) / 72.0 + 0.5 / 72.0 for x in rect]
    ax = matplotlib.pyplot.axes(rect)
    axv = matplotlib.pyplot.axes(rect, facecolor = None, frameon = False, sharex = ax)

    lines = []
    for k in sma.keys():
        if skip_weekends:
            # Plot the lines in indices; will replace the tics with custom label later
            sma_line = matplotlib.lines.Line2D(quotes[:N, 0], sma[k][:N], label = 'SMA ' + str(k))
        else:
            sma_line = matplotlib.lines.Line2D(quotes[:N, 1], sma[k][:N], label = 'SMA ' + str(k))
        lines.append(sma_line)
        ax.add_line(sma_line)
        if np.sum(np.isfinite(sma[k][:N])):
            y = np.nanmin(sma[k][:N])
            if y < ylim[0]:
                ylim[0] = y
            y = np.nanmax(sma[k][:N])
            if y > ylim[1]:
                ylim[1] = y

    # Round toward nice numbers
    if ylim[1] < 10:
        ylim[0] = np.floor(ylim[0])
        ylim[1] = np.ceil(ylim[1] * 2.0 + 1.0) * 0.5
    else:
        ylim[0] = np.floor(ylim[0] * 0.2) * 5.0
        ylim[1] = np.ceil(ylim[1] * 0.2) * 5.0

    def candlestick(ax, quotes, width = 0.5, linewidth = 1.0, volume_axis = None, skip_weekends = True, colormap = colorscheme.colorscheme('sunrise')):
        linewidth = 1.0
        offset = 0.7 * width;

        majors = []
        vlines = []
        olines = []
        clines = []
        vrects = []
        for q in quotes:
            if skip_weekends:
                i, t, o, h, l, c = q[:6]
                # Gather the indices of weeday == Monday
                if matplotlib.dates.num2date(t).weekday() == 0:
                   majors.append(i)
            else:
                k, i, o, h, l, c = q[:6]
            if c >= o:
                color = colormap.up
            else:
                color = colormap.down
            vline = matplotlib.lines.Line2D(xdata = (i, i), ydata = (l, h), color = color, linewidth = linewidth)
            oline = matplotlib.lines.Line2D(xdata = (i + offset, i), ydata = (o, o), color = color, linewidth = linewidth)
            cline = matplotlib.lines.Line2D(xdata = (i - offset, i), ydata = (c, c), color = color, linewidth = linewidth)
            vlines.append(vline)
            olines.append(oline)
            clines.append(cline)
            ax.add_line(vline)
            ax.add_line(oline)
            ax.add_line(cline)

        if volume_axis != None:
            for q in quotes:
                if skip_weekends:
                    i, t, o, h, l, c, v = q[:7]
                else:
                    k, i, o, h, l, c, v = q[:7]
                if c >= o:
                    color = colormap.bar_up
                else:
                    color = colormap.bar_down
                vrect = matplotlib.patches.Rectangle(xy = (i - 0.5, 0.0),
                    fill = True,
                    width = 1.0,
                    height = v,
                    facecolor = color,
                    edgecolor = colormap.text,
                    linewidth = 0.75,
                    alpha = 0.33)
                vrects.append(vrect)
                volume_axis.add_patch(vrect)

        N = len(quotes)

        def format_date(x, pos = None):
            index = int(x)
            if x < 0:
                #print('Project to {} days from {}.'.format(-index, matplotlib.dates.num2date(quotes[0, 1]).strftime('%b %d')))
                k = 0
                t = quotes[0, 1]
                while (k <= -index):
                    t = t - 1.0
                    weekday = matplotlib.dates.num2date(t).weekday()
                    # Only count Mon through Friday
                    if weekday >= 0 and weekday <= 4:
                        k = k + 1
                    #print('index = {}   weekday {}   k = {}'.format(index, weekday, k))
                date = matplotlib.dates.num2date(t)
                #print('date -> {}'.format(date))
            elif index > N - 1:
                return ''
            else:
                date = matplotlib.dates.num2date(quotes[index, 1])
            # print('x = {}   index = {} --> {} ({})'.format(x, index, date.strftime('%b %d'), date.weekday()))
            return date.strftime('%b %d')

        if skip_weekends:
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))
            ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1.0))
            # ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors))
            ax.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(base = 5.0, offset = majors[0]))  # Use the latest Monday
        else:
            mondays = matplotlib.dates.WeekdayLocator(matplotlib.dates.MONDAY)      # major ticks on the mondays
            alldays = matplotlib.dates.DayLocator()                                 # minor ticks on the days
            ax.xaxis.set_major_locator(mondays)
            ax.xaxis.set_minor_locator(alldays)
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))

    candlestick(ax, quotes[:N], volume_axis = axv, skip_weekends = skip_weekends, colormap = colormap)

    # Backdrop gradient
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('backdrop', colormap.backdrop)
    if skip_weekends:
        extent = [N, -1, ylim[0], ylim[1]]
    else:
        extent = [tt[N], tt[0] + 1, ylim[0], ylim[1]]
    ax.imshow(np.linspace(0, 1, 100).reshape(-1, 1), extent = extent, aspect = 'auto', cmap = cmap)

    matplotlib.pyplot.setp(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

    lines[0].set_color(colormap.line[0])
    lines[1].set_color(colormap.line[1])
    lines[2].set_color(colormap.line[2])

    leg = ax.legend(handles = lines, loc = 'best', facecolor = colormap.background, framealpha = 0.9)
    for text in leg.get_texts():
        text.set_color(colormap.text)
    ax.grid(color = colormap.grid, linestyle=':')
    ax.set_ylim(ylim)
    ax.yaxis.tick_right()
    ax.spines['top'].set_color(colormap.text)
    ax.spines['bottom'].set_color(colormap.text)
    ax.spines['left'].set_color(colormap.text)
    ax.spines['right'].set_color(colormap.text)
    ax.tick_params(axis = 'x', which = 'both', colors = colormap.text)
    ax.tick_params(axis = 'y', which = 'both', colors = colormap.text)
    axv.tick_params(axis = 'x', which = 'both', colors = colormap.text)
    axv.tick_params(axis = 'y', which = 'both', colors = colormap.text)

    ax.set_title('', color = colormap.text)

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

    # Compute the RSI curve
    rsi = RSI(dat.loc[:, 'Close'], rsi_period)

    # RSI axis and volume axis
    rect = RSI_RECT
    rect = [round(x * 72.0) / 72.0 + 0.5 / 72.0 for x in rect]
    axr = matplotlib.pyplot.axes(rect)
    rsi_line = matplotlib.lines.Line2D(xdata = (i, i), ydata = (l, h), color = color, linewidth = linewidth)
    if skip_weekends:
        # Plot the lines in indices; will replace the tics with custom label later
        rsi_line = matplotlib.lines.Line2D(quotes[:N, 0], rsi[-N:], label = 'RSI ' + str(rsi_period))
    else:
        rsi_line = matplotlib.lines.Line2D(quotes[:N, 1], rsi[-N:], label = 'RSI ' + str(rsi_period))

    dic = {'figure':fig, 'axes':ax, 'lines':lines, 'volume_axis':axv, 'rsi_axis':axr, 'close':matplotlib.pyplot.close}
    return dic

# def update(obj, quotes):

class Chart:
    """
        A chart class
    """
    def __init__(self, n, data = None, sma_sizes = DEFAULT_SMA_SIZES, color_scheme = 'sunrise', skip_weekends = True, forecast = 0):
        linewidth = 1.0
        width = 0.5
        offset = 0.4

        self.n = n
        self.sma = dict.fromkeys(sma_sizes)
        self.symbol = ''
        self.colormap = colorscheme.colorscheme(color_scheme)
        self.skip_weekends = skip_weekends
        self.forecast = forecast

        self.fig = matplotlib.pyplot.figure()
        self.fig.patch.set_alpha(0.0)
        # dpi = self.fig.dpi
        # print('dpi = {}'.format(dpi))
        # self.rect = [(round(x * self.dpi)  + 0.5) / dpi for x in rect]
        # self.rect = MAIN_RECT
        self.axb = self.fig.add_axes(BACK_RECT, frameon = False)
        self.axb.yaxis.set_visible(False)
        self.axb.xaxis.set_visible(False)
        
        self.axq = self.fig.add_axes(MAIN_RECT, label = 'Quotes')
        self.axq.patch.set_visible(False)
        self.axq.yaxis.tick_right()
        
        self.axv = self.fig.add_axes(MAIN_RECT, frameon = False, sharex = self.axq)
        self.axv.patch.set_visible(False)
        self.axv.xaxis.set_visible(False)
        
        # Backdrop gradient
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('backdrop', self.colormap.backdrop)
        fprop = matplotlib.font_manager.FontProperties(style = 'normal', size = 60, weight = 'bold', stretch = 'normal')
        self.im = self.axb.imshow(np.linspace(0, 1, 100).reshape(-1, 1), cmap = cmap, extent = (-1, 1, -1, 1), aspect = 'auto')
        self.st = self.axb.text(0, 0, self.symbol,
            fontproperties = fprop, horizontalalignment = 'center', verticalalignment = 'center',
            color = self.colormap.background_text_color, alpha = self.colormap.background_text_alpha)

        # SMA lines
        self.lines = []
        for j, k in enumerate(self.sma.keys()):
            sma_line = matplotlib.lines.Line2D(range(self.n), np.multiply(range(self.n), k / self.n), label = 'SMA ' + str(k), color = self.colormap.line[j])
            self.axq.add_line(sma_line)
            self.lines.append(sma_line)

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
                width = 1.0,
                height = 10.0,
                facecolor = '#0000ff',
                edgecolor = self.colormap.text,
                linewidth = 0.75,
                alpha = 0.33)
            self.vlines.append(vline)
            self.olines.append(oline)
            self.clines.append(cline)
            self.vrects.append(vrect)
            self.axq.add_line(vline)
            self.axq.add_line(oline)
            self.axq.add_line(cline)
            self.axv.add_patch(vrect)            

        # A forecast point
        line = matplotlib.lines.Line2D(xdata = (self.n + 10.0, self.n + 10.0), ydata = (100.0, 100.0), color = 'r')
        self.axq.add_line(line)

        # Legend
        self.leg = self.axq.legend(handles = self.lines, loc = 'upper left', facecolor = self.colormap.background, framealpha = 0.85)
        for text in self.leg.get_texts():
            text.set_color(self.colormap.text)

        # Grid
        self.axq.grid(color = self.colormap.grid, linestyle=':')
        self.axq.spines['top'].set_visible(False)
        self.axv.spines['top'].set_visible(False)
        for side in ['bottom', 'left', 'right']:
            self.axq.spines[side].set_color(self.colormap.text)
            self.axv.spines[side].set_color(self.colormap.text)
        self.axq.tick_params(axis = 'x', which = 'both', colors = self.colormap.text)
        self.axq.tick_params(axis = 'y', which = 'both', colors = self.colormap.text)
        self.axv.tick_params(axis = 'x', which = 'both', colors = self.colormap.text)
        self.axv.tick_params(axis = 'y', which = 'both', colors = self.colormap.text)

        self.axq.xaxis.set_minor_locator(matplotlib.ticker.IndexLocator(1, 0))

        self.axq.set_xlim([-1.5, self.n + 0.5])
        self.axv.set_xlim([-1.5, self.n + 0.5])
        self.axq.xaxis.set_data_interval(-1.0, self.n + 10.0)
        self.axv.xaxis.set_data_interval(-1.0, self.n + 10.0)

        self.axq.set_ylim([0, 110])
        self.axv.set_ylim([0, 10])
        self.axr = self.fig.add_axes(RSI_RECT, label = 'RSI')
        self.axr.patch.set_visible(False)
        self.axr.xaxis.set_visible(False)

        # self.axr.spines['bottom'].set_color(self.colormap.grid)
        self.axr.spines['bottom'].set_visible(False)
        self.axr.tick_params(axis = 'x', which = 'both', colors = self.colormap.text)
        self.axr.tick_params(axis = 'y', which = 'both', colors = self.colormap.text)

        self.title = self.axr.set_title(self.symbol, color = self.colormap.text)

        if data is not None:
            self.set_xdata(data.major_axis)
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
        n = majors[-1]
        #print('DEBUG: {}, ..., {} -> \033[38;5;214m{}\033[0m ({})'.format(majors[0], n, dates[n].strftime('%Y-%m-%d'), dates[n].weekday()))

        def format_date(x, pos = None):
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
                        k = k + 1
                    #print('index = {}   weekday {}   k = {}'.format(index, weekday, k))
                date = matplotlib.dates.num2date(t)
                #print('date -> {}'.format(date))
            elif index > self.n - 1:
                #print('Extrapolate {} days from {}.'.format(index - self.n, matplotlib.dates.num2date(dates[-1]).strftime('%b %d')))
                k = 0
                t = dnums[-1]
                while (k <= index - self.n):
                    t = t + 1.0
                    weekday = matplotlib.dates.num2date(t).weekday()
                    # Only count Mon through Friday
                    if weekday >= 0 and weekday <= 4:
                        k = k + 1
                    #print('index = {}   weekday {}   k = {}'.format(index, weekday, k))
                date = matplotlib.dates.num2date(t)
            else:
                date = dates[index]
            #print('x = {}   index = {} --> {} ({})'.format(x, index, date.strftime('%b %d'), date.weekday()))
            return date.strftime('%b %d')

        if self.skip_weekends:
            #print((self.n - majors[-1]) + 1)
            self.axq.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))
            self.axq.xaxis.set_minor_locator(matplotlib.ticker.IndexLocator(1, 0))
            # self.axq.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
            # self.axq.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors))
            # self.axq.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(5, (self.n - majors[-1]) - 1))  # Use the last Monday
            self.axq.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(5, majors[-1] % 5 + 1))  # Use the last Monday
        else:
            mondays = matplotlib.dates.WeekdayLocator(matplotlib.dates.MONDAY)      # major ticks on the mondays
            alldays = matplotlib.dates.DayLocator()                                 # minor ticks on the days
            self.axq.xaxis.set_major_locator(mondays)
            self.axq.xaxis.set_minor_locator(alldays)
            self.axq.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))

        matplotlib.pyplot.setp(self.axq.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

    def set_data(self, panel):
        # Get the symbol from minor axis
        self.symbol = panel.minor_axis[0]

        # Get the first frame
        data = panel.loc[:, :, self.symbol]
        quotes = np.transpose([
            data.loc[:, 'Open'].tolist(),
            data.loc[:, 'High'].tolist(),
            data.loc[:, 'Low'].tolist(),
            data.loc[:, 'Close'].tolist(),
            np.multiply(data.loc[:, 'Volume'], 1.0e-6).tolist()
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
            sma = np.convolve(quotes[:, 3], np.ones((k, )) / k, mode = 'valid')
            if len(sma) < self.n:
                self.sma[k] = np.concatenate((np.full(self.n - len(sma), np.nan), sma))
            else:
                self.sma[k] = sma[-self.n:]
            self.lines[j].set_ydata(self.sma[k])
            # self.axq.draw_artist(self.lines[j])
            if np.sum(np.isfinite(self.sma[k])):
                qlim[0] = min(qlim[0], np.nanpercentile(self.sma[k], 25))
                qlim[1] = max(qlim[1], np.nanpercentile(self.sma[k], 75))
        if qlim[1] - qlim[0] < 10:
            qlim = [round(qlim[0]) - 0.5, round(qlim[1]) + 0.5]
        else:
            qlim = [round(qlim[0] * 0.2 - 1.0) * 5.0, round(qlim[1] * 0.2 + 1.0) * 5.0]

        # Legend position: upper right if SMA-N is increasing, upper left otherwise (Not in public API)
        sma = self.sma[list(self.sma)[-2]]
        if sma[-10] > sma[0]:
            self.leg._loc = 2
        else:
            self.leg._loc = 1

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

