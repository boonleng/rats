import math
import numpy as np
import matplotlib
import matplotlib.pyplot

def candlestick(ax, quotes, width = 0.4, linewidth = 1.0, volume_axis = None, weekends = False):
    linewidth = 1.0
    offset = 0.5 * width;
    
    # Original dates
    tt = list(quotes[:, 1])
    N = len(tt)
    #quotes[:, 0] = np.arange(0, -N, -1)

    majors = []
    vlines = []
    olines = []
    clines = []
    vrects = []
    for q in quotes:
        if weekends:
            k, i, o, h, l, c = q[:6]
        else:
            i, t, o, h, l, c = q[:6]
            # Gather the indices of weeday == 1 ()
            if matplotlib.dates.num2date(t).weekday() == 1:
               majors.append(i)
        if c >= o:
            color = 'k'
        else:
            color = 'r'
        vline = matplotlib.lines.Line2D(xdata = (i, i), ydata = (l, h), color = color, linewidth = linewidth)
        oline = matplotlib.lines.Line2D(xdata = (i - offset, i), ydata = (o, o), color = color, linewidth = linewidth)
        cline = matplotlib.lines.Line2D(xdata = (i + offset, i), ydata = (c, c), color = color, linewidth = linewidth)
        vlines.append(vline)
        olines.append(oline)
        clines.append(cline)
        ax.add_line(vline)
        ax.add_line(oline)
        ax.add_line(cline)

    if volume_axis != None:
        for q in quotes:
            if weekends:
                k, i, o, h, l, c, v = q[:7]
            else:
                i, t, o, h, l, c, v = q[:7]
            if c >= o:
                color = 'g'
            else:
                color = 'r'
            vrect = matplotlib.patches.Rectangle(xy = (i - offset, 0.0),
                fill = True,
            	width = 2.5 * width,
            	height = v,
            	facecolor = color,
            	edgecolor = 'k',
            	linewidth = 0.5,
            	alpha = 0.3)
            vrects.append(vrect)
            volume_axis.add_patch(vrect)

    ax.autoscale_view()

    def format_date(x, pos = None):
        index = int(0.5 + x)
        if index < 0 or index > N - 1:
            return ""
        else:
            date = matplotlib.dates.num2date(quotes[index, 1])
        #print('x = {}   index = {} --> {} ({})'.format(x, int(0.5 - x), date.strftime('%b %d'), date.weekday()))
        return date.strftime('%b %d')

    if weekends:
        mondays = matplotlib.dates.WeekdayLocator(matplotlib.dates.MONDAY)      # major ticks on the mondays
        alldays = matplotlib.dates.DayLocator()                                 # minor ticks on the days
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
    else:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1.0))
        # ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(majors))
        ax.xaxis.set_major_locator(matplotlib.ticker.IndexLocator(base = 5.0, offset = majors[0]))  # Use the latest Monday

def showChart(dat, sma_sizes = [10, 20, 50], weekends = False):
    fig = matplotlib.pyplot.figure()
    fig.patch.set_alpha(0.0)
    rect = [0.075, 0.12, 0.83, 0.78]
    rect = [round(x * 72.0) / 72.0 + 0.5 / 72.0 for x in rect]

    ii = list(range(len(dat)))
    tt = list(matplotlib.dates.date2num(dat.index.tolist()))
    oo = dat.loc[:, "Open"].tolist()
    hh = dat.loc[:, "High"].tolist()
    ll = dat.loc[:, "Low"].tolist()
    cc = dat.loc[:, "Close"].tolist()
    vv = np.multiply(dat.loc[:, "Volume"], 1.0e-6).tolist()
    quotes = np.transpose([ii, tt, oo, hh, ll, cc, vv])

    # Initialize an empty dictionary from keys based on sma size
    sma = dict.fromkeys(sma_sizes)
    n = 0
    for num in sma.keys():
        n = max(n, num)

    # Prepare the SMA curves
    N = len(dat) - n - 1
    for k in sma.keys():
        sma[k] = np.convolve(cc, np.ones((k, )) / k, mode = 'valid')
        sma[k] = np.pad(sma[k], (0, k - 1), mode = 'constant', constant_values = np.nan)

    nums = np.array([oo, ll, hh, cc])
    ylim = [nums[:, :N].flatten().min(), nums[:, :N].flatten().max()]

    ax = matplotlib.pyplot.axes(rect)
    axv = matplotlib.pyplot.axes(rect, facecolor = None, frameon = False, sharex = ax)

    lines = []
    for k in sma.keys():
        if weekends:
            sma_line = matplotlib.lines.Line2D(tt[:N], sma[k][:N], label = 'SMA ' + str(k))
        else:
            # Plot the lines in indices; will replace the tics with custom label later
            sma_line = matplotlib.lines.Line2D(ii[:N], sma[k][:N], label = 'SMA ' + str(k))
        lines.append(sma_line)
        ax.add_line(sma_line)
        ylim[0] = min(ylim[0], np.nanmin(sma[k][:N]))
        ylim[1] = max(ylim[1], np.nanmax(sma[k][:N]))
    ylim[0] = math.floor(ylim[0] * 0.2) * 5.0
    ylim[1] = math.ceil(ylim[1] * 0.2 + 1) * 5.0

    candlestick(ax, quotes[:N], volume_axis = axv, weekends = weekends)

    # Backdrop
    # shades = ['#c9e6e3', '#ffe6a9', '#ebc3bc']
    shades = ['#ffffbb', '#ffcccc', '#ccccff']
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('sunset', shades)
    if weekends:
        extent = [tt[N], tt[0] + 10, ylim[0], ylim[1]]
    else:
        extent = [N, -10, ylim[0], ylim[1]]
    ax.imshow(np.linspace(0, 1, 100).reshape(-1, 1),
               extent = extent,
               aspect = 'auto',
               cmap = cmap)

    matplotlib.pyplot.setp(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

    lines[0].set_color('#1155ff')
    lines[1].set_color('#ee8822')
    lines[2].set_color('#559900')

    ax.legend(handles = lines, loc = 2)
    ax.grid(linestyle=':')
    ax.set_ylim(ylim)
    ax.yaxis.tick_right()

    # Volume bars to have the mean at around 10% of the vertical space
    blim = [0, math.ceil(np.array(vv).mean()) * 10]

    axv.set_ylim(blim)
    yticks = axv.get_yticks()
    new_ticks = []
    new_labels = []
    for ii in range(1, math.ceil(len(yticks) / 3)):
        new_ticks.append(yticks[ii])
        new_labels.append(str(math.floor(yticks[ii])) + 'M')
    axv.set_yticks(new_ticks)
    axv.set_yticklabels(new_labels)
    axv.xaxis.set_visible(False)

    dic = {'figure':fig, 'axes':ax, 'lines':lines, 'volume_axis':axv}
    return dic
