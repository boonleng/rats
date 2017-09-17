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
