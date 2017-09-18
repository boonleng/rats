import numpy as np
import matplotlib
import matplotlib.pyplot
import colorscheme

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

def showChart(dat, sma_sizes = [10, 50, 100], skip_weekends = True, color_scheme = 'sunrise'):
    """
        showChart(dat, sma_size = [10, 20, 50], skip_weekends = True):)
        - dat - Data frame from pandas-datareader
        - sma_sizes - Window sizes for SMA (sliding moving average)
        - skip_weekends - Skip plotting weekends and days with no data
    """
    fig = matplotlib.pyplot.figure()
    fig.patch.set_alpha(0.0)
    rect = [0.075, 0.12, 0.83, 0.78]
    rect = [round(x * 72.0) / 72.0 + 0.5 / 72.0 for x in rect]

    colormap = colorscheme.colorscheme(color_scheme)

    ii = list(range(len(dat)))
    tt = list(matplotlib.dates.date2num(dat.index.tolist()))
    oo = dat.loc[:, "Open"].tolist()
    hh = dat.loc[:, "High"].tolist()
    ll = dat.loc[:, "Low"].tolist()
    cc = dat.loc[:, "Close"].tolist()
    vv = np.multiply(dat.loc[:, "Volume"], 1.0e-6).tolist()
    quotes = np.transpose([ii, tt, oo, hh, ll, cc, vv])

    # Sort the data  (colums 1 ... 6) so that it is newest first
    if tt[1] > tt[0]:
        # print('Resorting ... {}  {}'.format(quotes.shape, sma_sizes))
        quotes[:, 1:8] = quotes[::-1, 1:8]

    # Initialize an empty dictionary from keys based on sma size
    sma = dict.fromkeys(sma_sizes)
    n = 0
    for num in sma.keys():
        n = max(n, num)

    # Prepare the SMA curves
    N = len(dat) - n - 1
    # print('N = {} - {} - 1 = {}'.format(len(dat), n, N))
    for k in sma.keys():
        sma[k] = np.convolve(quotes[:, 5], np.ones((k, )) / k, mode = 'valid')
        sma[k] = np.pad(sma[k], (0, k - 1), mode = 'constant', constant_values = np.nan)

    # Find the span of colums 2 to 5 (OHLC)
    nums = np.array(quotes[:N, 2:6]).flatten()
    ylim = [np.nanmin(nums), np.nanmax(nums)]

    # Main axis and volume axis
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

    candlestick(ax, quotes[:N], volume_axis = axv, skip_weekends = skip_weekends, colormap = colormap)

    # Backdrop gradient
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('backdrop', colormap.backdrop)
    if skip_weekends:
        extent = [N, -10, ylim[0], ylim[1]]
    else:
        extent = [tt[N], tt[0] + 10, ylim[0], ylim[1]]
    ax.imshow(np.linspace(0, 1, 100).reshape(-1, 1), extent = extent, aspect = 'auto', cmap = cmap)

    matplotlib.pyplot.setp(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

    lines[0].set_color(colormap.line[0])
    lines[1].set_color(colormap.line[1])
    lines[2].set_color(colormap.line[2])

    lge = ax.legend(handles = lines, loc = 'best', facecolor = colormap.background, framealpha = 0.9)
    for text in lge.get_texts():
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
    v = np.nanmean(np.array(vv))
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

    dic = {'figure':fig, 'axes':ax, 'lines':lines, 'volume_axis':axv}
    return dic

# def update(obj, quotes):

class Chart:
    """
        A chart class
    """
    def __init__(self, n, sma_sizes = [10, 50, 100], color_scheme = 'sunrise', skip_weekends = True):
        linewidth = 1.0
        width = 0.5
        offset = 0.4
        rect = [0.075, 0.14, 0.83, 0.78]

        self.n = n
        self.sma = dict.fromkeys(sma_sizes)
        self.colormap = colorscheme.colorscheme(color_scheme)
        self.skip_weekends = skip_weekends

        self.fig = matplotlib.pyplot.figure()
        self.fig.patch.set_alpha(0.0)
        #dpi = self.fig.dpi
        #print('dpi = {}'.format(dpi))
        #self.rect = [(round(x * self.dpi)  + 0.5) / dpi for x in rect]
        self.rect = rect
        self.axb = self.fig.add_axes(self.rect, frameon = False)
        self.axb.yaxis.set_visible(False)
        self.axb.xaxis.set_visible(False)
        
        self.axq = self.fig.add_axes(self.rect, label = 'Quotes')
        self.axq.patch.set_visible(False)
        self.axq.yaxis.tick_right()
        
        self.axv = self.fig.add_axes(self.rect, frameon = False, sharex = self.axq)
        self.axv.patch.set_visible(False)
        self.axv.xaxis.set_visible(False)
        
        # Backdrop gradient
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('backdrop', self.colormap.backdrop)
        self.im = self.axb.imshow(np.linspace(0, 1, 100).reshape(-1, 1), cmap = cmap, aspect = 'auto')

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
                height = (i % 10) * 0.5,
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
        self.leg = self.axq.legend(handles = self.lines, loc = 'best', facecolor = self.colormap.background, framealpha = 0.9)
        for text in self.leg.get_texts():
            text.set_color(self.colormap.text)

        # Grid
        self.axq.grid(color = self.colormap.grid, linestyle=':')
        for side in ['top', 'bottom', 'left', 'right']:
            self.axq.spines[side].set_color(self.colormap.text)
            self.axv.spines[side].set_color(self.colormap.text)
        self.axq.tick_params(axis = 'x', which = 'both', colors = self.colormap.text)
        self.axq.tick_params(axis = 'y', which = 'both', colors = self.colormap.text)
        self.axv.tick_params(axis = 'x', which = 'both', colors = self.colormap.text)
        self.axv.tick_params(axis = 'y', which = 'both', colors = self.colormap.text)

        self.axq.xaxis.set_minor_locator(matplotlib.ticker.IndexLocator(1, 0))

        self.axq.set_xlim([-1.5, self.n + 9.5])
        self.axv.set_xlim([-1.5, self.n + 9.5])
        self.axq.xaxis.set_data_interval(-1.0, self.n + 10.0)
        self.axv.xaxis.set_data_interval(-1.0, self.n + 10.0)

        self.axq.set_ylim([0, 110])
        self.axv.set_ylim([0, 10])

        self.title = self.axq.set_title('', color = self.colormap.text)

    def set_xdata(self, xdata):
        if len(xdata) != self.n:
            print('xdata must have the same length as the chart setup (n = {}).'.format(self.n))
            return
        dates = matplotlib.dates.date2num(xdata.tolist())

        # Gather the major indices of weeday == Monday
        majors = []
        for i, t in enumerate(dates):
            if matplotlib.dates.num2date(t).weekday() == 0:
               majors.append(i)

        def format_date(x, pos = None):
            index = int(x)
            # print('x = {}'.format(x))
            if x < 0:
                #print('Project to {} days from {}.'.format(-index, matplotlib.dates.num2date(dates[0]).strftime('%b %d')))
                k = 0
                t = dates[0]
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
                t = dates[-1]
                while (k <= index - self.n):
                    t = t + 1.0
                    weekday = matplotlib.dates.num2date(t).weekday()
                    # Only count Mon through Friday
                    if weekday >= 0 and weekday <= 4:
                        k = k + 1
                    #print('index = {}   weekday {}   k = {}'.format(index, weekday, k))
                date = matplotlib.dates.num2date(t)
            else:
                date = matplotlib.dates.num2date(dates[index])
            #print('x = {}   index = {} --> {} ({})'.format(x, index, date.strftime('%b %d'), date.weekday()))
            return date.strftime('%b %d')

        if self.skip_weekends:
            #print((self.n - majors[-1]) + 1)
            self.axq.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))
            self.axq.xaxis.set_minor_locator(matplotlib.ticker.IndexLocator(1, 0))
            # self.axq.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
            # self.axq.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors))
            self.axq.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(5, (self.n - majors[-1]) + 1))  # Use the last Monday
        else:
            mondays = matplotlib.dates.WeekdayLocator(matplotlib.dates.MONDAY)      # major ticks on the mondays
            alldays = matplotlib.dates.DayLocator()                                 # minor ticks on the days
            self.axq.xaxis.set_major_locator(mondays)
            self.axq.xaxis.set_minor_locator(alldays)
            self.axq.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))

        matplotlib.pyplot.setp(self.axq.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

    def set_data(self, data):
        quotes = np.transpose([
            data.loc[:, "Open"].tolist(),
            data.loc[:, "High"].tolist(),
            data.loc[:, "Low"].tolist(),
            data.loc[:, "Close"].tolist(),
            np.multiply(data.loc[:, "Volume"], 1.0e-6).tolist()
        ])
        for k, q in enumerate(quotes[-self.n:, :]):
            o, h, l, c, v = q[:5]
            if c >= o:
                line_color = self.colormap.up
                bar_color = self.colormap.bar_up
            else:
                line_color = self.colormap.down
                bar_color = self.colormap.bar_down
            self.vlines[k].set_ydata((l, h))
            self.vlines[k].set_color(line_color)
            self.olines[k].set_ydata((o, o))
            self.olines[k].set_color(line_color)
            self.clines[k].set_ydata((c, c))    
            self.clines[k].set_color(line_color)
            self.vrects[k].set_height(v)
            self.vrects[k].set_facecolor(bar_color)

        # Compute SMA
        for j, k in enumerate(self.sma.keys()):
            sma = np.convolve(quotes[:, 3], np.ones((k, )) / k, mode = 'valid')
            self.sma[k] = sma[-self.n:]
            self.lines[j].set_ydata(self.sma[k])

        # Find the span of colums (OHLC)
        nums = np.array(quotes[-self.n:, 0:4]).flatten()
        ylim = [round(np.nanmin(nums) * 0.2 - 1.0) * 5.0, round(np.nanmax(nums) * 0.2 + 1.0) * 5.0]
        self.axq.set_ylim(ylim)

        # Volume bars to have the mean at around 10% of the vertical space
        v = np.nanmean(quotes[-self.n:, 4])
        new_ticks = []
        new_labels = []
        blim = [0, np.ceil(v * 10.0)]
        if v < 1.0:
            steps = np.array([0.02, 0.05, 0.1, 0.2, 0.5])
            step = steps[np.argmin(np.abs(steps - v))]
            t = 0.0
            while t < 3.0 * v:
                t = t + step
                new_ticks.append(t)
                new_labels.append(str(int(t * 100)) + 'K')
        else:
            steps = np.array([1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500])
            step = steps[np.argmin(np.abs(steps - v))]
            t = 0.0
            while t < 3.0 * v:
                t = t + step
                new_ticks.append(t)
                new_labels.append(str(int(t)) + 'M')
        # print('step = {}'.format(step))
        self.axv.set_ylim(blim)
        self.axv.set_yticks(new_ticks)
        self.axv.set_yticklabels(new_labels)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def set_title(self, title):
        self.title.set_text(title)

    def savefig(self, filename):
        self.fig.savefig(filename)

