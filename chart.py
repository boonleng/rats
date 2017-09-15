import math
import numpy as np
import matplotlib
import matplotlib.pyplot

def candlestick(ax, quotes, width = 0.5, linewidth = 1.0, volume_axis = None, skip_weekends = True):
    linewidth = 1.0
    
    # Original dates
    tt = list(quotes[:, 1])
    N = len(tt)
    #quotes[:, 0] = np.arange(0, -N, -1)

    daterev = tt[0] - tt[1] > 0
    if daterev:
        offset = -0.5 * width;
    else:
        offset = 0.5 * width;

    majors = []
    vlines = []
    olines = []
    clines = []
    vrects = []
    for q in quotes:
        if skip_weekends:
            i, t, o, h, l, c = q[:6]
            # Gather the indices of weeday == 1 ()
            if matplotlib.dates.num2date(t).weekday() == 1:
               majors.append(i)
        else:
            k, i, o, h, l, c = q[:6]
        if c >= o:
            color = 'k'
            #color = '#33ff00'
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
            if skip_weekends:
                i, t, o, h, l, c, v = q[:7]
            else:
                k, i, o, h, l, c, v = q[:7]
            if c >= o:
                color = 'g'
                # color = '#33ff00'
            else:
                color = 'r'
            vrect = matplotlib.patches.Rectangle(xy = (i - 0.5, 0.0),
                fill = True,
            	width = 1.0,
            	height = v,
            	facecolor = color,
            	edgecolor = 'k',
            	linewidth = 0.75,
            	alpha = 0.33)
            vrects.append(vrect)
            volume_axis.add_patch(vrect)

    ax.autoscale_view()

    def format_date(x, pos = None):
        index = int(0.5 + x)
        if index < 0:
            #print('Project to {} days from {}.'.format(-index, matplotlib.dates.num2date(quotes[0, 1]).strftime('%b %d')))
            k = 0
            t = quotes[0, 1]
            while (k <= -index):
                if daterev:
                    t = t + 1.0
                else:
                    t = t - 1.0
                # date = matplotlib.dates.num2date(quotes[0, 1] + k)
                weekday = matplotlib.dates.num2date(t).weekday()
                # Only count Mon - Friday
                if weekday > 0 and weekday < 6:
                    k = k + 1
                #print('k = {}  day {}'.format(k, weekday))
            date = matplotlib.dates.num2date(t)
        elif index > N - 1:
            return ""
        else:
            date = matplotlib.dates.num2date(quotes[index, 1])
        #print('x = {}   index = {} --> {} ({})'.format(x, index, date.strftime('%b %d'), date.weekday()))
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

def showChart(dat, sma_sizes = [10, 20, 50], skip_weekends = True):
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
    ylim = [np.nanmin(nums[:, :N].flatten()), np.nanmax(nums[:, :N].flatten())]

    # Main axis and volume axis
    ax = matplotlib.pyplot.axes(rect)
    axv = matplotlib.pyplot.axes(rect, facecolor = None, frameon = False, sharex = ax)

    lines = []
    for k in sma.keys():
        if skip_weekends:
            # Plot the lines in indices; will replace the tics with custom label later
            sma_line = matplotlib.lines.Line2D(ii[:N], sma[k][:N], label = 'SMA ' + str(k))
        else:
            sma_line = matplotlib.lines.Line2D(tt[:N], sma[k][:N], label = 'SMA ' + str(k))
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

    candlestick(ax, quotes[:N], volume_axis = axv, skip_weekends = skip_weekends)

    # Backdrop
    # shades = ['#c9e6e3', '#ffe6a9', '#ebc3bc']
    # shades = ['#ffffbb', '#ffcccc', '#ccccff']
    shades = ['#ccccff', '#d8ccea', '#e5cce5', '#f8cccc', '#fbdecc', '#fff0bb', '#f0f0ee']
    # shades = ['#000033', '#003366']
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('sunset', shades)
    if skip_weekends:
        extent = [N, -10, ylim[0], ylim[1]]
    else:
        extent = [tt[N], tt[0] + 10, ylim[0], ylim[1]]
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
    v = np.nanmean(np.array(vv))
    if v < 1.0:
        blim = [0, math.ceil(v * 10.0)]
    else:
        blim = [0, math.ceil(v) * 10.0]
    axv.set_ylim(blim)
    yticks = axv.get_yticks()
    new_ticks = []
    new_labels = []
    for ii in range(1, math.ceil(len(yticks) / 3)):
        new_ticks.append(yticks[ii])
        if yticks[1] < 1.0:
            new_labels.append(str(math.floor(yticks[ii] * 100.0) * 10) + 'K')
        else:
            new_labels.append(str(math.floor(yticks[ii])) + 'M')
    axv.set_yticks(new_ticks)
    axv.set_yticklabels(new_labels)
    axv.xaxis.set_visible(False)

    dic = {'figure':fig, 'axes':ax, 'lines':lines, 'volume_axis':axv}
    return dic
