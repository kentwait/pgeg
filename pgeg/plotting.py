from matplotlib import pyplot as plt
import numpy as np


def plot_ci_graph(y, x,
                  y_err_series=None, x_err_series=None,
                  colors=None, shapes=None, labels=None,
                  line=False, markersize=15, fill=True, grid=True, legend=True, legendloc='center',
                  ylim=None, xlim=None, xlabel='', ylabel='', figsize=(12,8)):
    """Construct a confidence interval plot given stacked-x, -y, -xerr, and -yerr arrays.

    Parameters
    ----------
    y : list of y datasets
        The function iterates over y datasets to draw a plot for each set of y values.
    x : list of x datasets
        Similar to y, the function assumes that it is looping over sets of x values and plots them for each iteration.
    x_err_series : list of arrays, optional (default = None)
        Each array in the list contains a column for the lower-, and upper bounds to draw error bars for x values.
        The length of each array (number of rows) in x_err_series should be equal to the length of each array in x.
    y_err_series : list of arrays, optional (default = None)
        Each array in the list contains a column for the lower-, and upper bounds to draw error bars for y values.
        The length of each array in y_err_series should be equal to the length of each aray in y.
    colors : list
        List contains heaxadecimal strings (for example: '#ffffff') of colors to be used to plot datasets.
    shapes : list
        List contains the string representation (for example: 'o') of marker shapes to be used for plotting datasets.
    labels : list
        Descriptive names for each dataset to be listed in the legend.
    line : bool
        Whether to draw a line connecting x data

    Returns
    -------
    matplotlib.axis

    """
    assert len(x) == len(y)
    x_array, y_array = map(np.array, [x,y])
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    line_char = '-' if line else ''
    colors = ['#ff5a00', '#055499', '#cc0000', '#00abbd'] if colors is None else colors
    shapes = ['o'] if shapes is None else shapes
    labels = ['data {}'.format(i) for i in range(1, len(x) + 1)] if labels is None else labels
    assert len(x) == len(labels)
    for i in range(x_array.shape[0]):
        color = colors[i] if i < len(colors) else colors[i%len(colors)]
        shape = shapes[i] if i < len(shapes) else shapes[i%len(shapes)]
        ax.errorbar(y=y_array[i], x=x_array[i],
                    yerr=y_err_series[i] if y_err_series is not None else None,
                    xerr=x_err_series[i] if x_err_series is not None else None,
                    marker=shape, linestyle=line_char, markersize=markersize,
                    markeredgecolor='#ffffff' if fill else color,
                    markerfacecolor=color if fill else 'none',
                    markeredgewidth=1,
                    ecolor=color, label=labels[i])
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid: ax.grid()

    if legend:
        # Adjust subplot
        box = ax.get_position()
        # Draw legend
        handles, labels = ax.get_legend_handles_labels()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        if legendloc == 'upper': anchor = (1,1)
        elif legendloc == 'center': anchor = (1,0.5)
        elif legendloc == 'lower': anchor = (1,0)
        else: raise Exception('legendloc must be "upper", "center", or "lower"')
        ax.legend(handles, labels, loc='{} left'.format(legendloc), bbox_to_anchor=anchor, numpoints=1, fontsize=16)

    return ax


def plot_heatmap(df, cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                 row_labels=None, col_labels=None, row_labelsize=14, col_labelsize=14,
                 row_title='', col_title='', row_titlesize=16, col_titlesize=16,
                 legend_label='', legend_labelsize=14,
                 figsize=(12,8), alpha=1.0):
    """Construct a heatmap from a DataFrame.

    Parameters
    ----------
    df : DatFrame
    cmap : MatPlotLib colormap string or object, optional (default = RdBu_r)
    vmin : float, optional (default = -0.5)
    vmax : float, optional (default = 0.5)
    row_labels : list, optional (default = None)
    col_labels : list, optional (default = None)
    row_labelsize : int or float, optional (default = 14)
    col_labelsize : int or float, optional (default = 14)
    row_title : str, optional (default = '')
    col_title : str, optional (default = '')
    row_titlesize : int or float, optional (default = 16)
    col_titlesize : int or float, optional (default = 16)
    legend_label : str, optional (default = '')
    legend_labelsize : int or float, optional (default = 14)
    figsize : tuple, optional (default = (12,8))
    alpha : float, optional (default = 1.0)

    Returns
    -------
    matplotlib.axis

    """
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(df, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)

    # configure layout
    fig.set_size_inches(*figsize)
    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(df.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.label_position = 'top'
    ax.xaxis.labelpad = 10 + col_titlesize

    ax.tick_params(axis='both', tick1On=False, tick2On=False)
    ax.tick_params(axis='both', tick1On=False, tick2On=False)
    if col_title: ax.set_xlabel(col_title, size=col_titlesize)
    if row_title: ax.set_ylabel(row_title, size=row_titlesize)

    row_labels = [i for i in range(df.shape[0])] if row_labels is None else row_labels
    col_labels = [i for i in range(df.shape[1])] if col_labels is None else col_labels
    ax.set_yticklabels(row_labels, minor=False, size=row_labelsize)
    ax.set_xticklabels(col_labels, minor=False, size=col_labelsize)

    cbar = plt.colorbar(heatmap)
    cbar.set_label(legend_label)
    cbar.ax.tick_params(labelsize=legend_labelsize)

    return ax


def error_bar_array(df, interval=0.95):
    """

    Parameters
    ----------
    df
    interval

    Returns
    -------

    """
    data = ci(df, ci=interval)
    return np.vstack([data['lb_dist'], data['ub_dist']])
