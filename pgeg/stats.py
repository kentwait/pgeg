from collections import OrderedDict
import numpy as np
import pandas as pd

def ci(df, ci=0.95, empirical=True):
    """Returns the mean, median, bounds and distance from mean of the specified confidence interval

    Parameters
    ----------
    df : DataFrame to be analyzed.
         Columns should contain categories and rows are replicates. Mean, median, and distributions
         are analyzed by reading the rows of each column.
    ci : float, optional (default = 0.95)
        Confidence interval
    empirical : bool, optional (default = True)
        Whether to return a member of the dataset or an interpolated estimate when the percentile rank does not
        correspond to a data point that is part of the dataset.

    Returns
    -------
    DataFrame
        Pandas DataFrame containing mean, median, lower and upper bound values of confidence interval, and
        lower and upper bound distances from the mean of the confidence interval

    """
    assert isinstance(df, pd.DataFrame)
    dist = (1-ci)/float(2)
    data = OrderedDict()
    data['mean'] = df.mean()
    data['median'] = df.median()
    if empirical:
        data['lb'] = np.percentile(df, dist*100, axis=0, interpolation='lower')
        data['ub'] = np.percentile(df, (1 - dist)*100, axis=0, interpolation='lower')
    else:
        data['lb'] = df.quantile(dist)
        data['ub'] = df.quantile(1 - dist)
    data['lb_dist'] = data['mean'] - data['lb']
    data['ub_dist'] = data['ub'] - data['mean']
    return pd.DataFrame(data)

def skew(a, b):
    """Computes the skew based on a and b DataFrames

    Skew is computed using the formula:

        skew = (a - b) / (a + b)
        where a, b are DataFrames

    The function expects that replicates are found in the rows (major axis) of the DataFrame and columns represent the
    different categories.

    Parameters
    ----------
    a : DataFrame
    b : DataFrame

    Returns
    DataFrame
        The Dataframe rows (major axis) are replicates while the columns (minor axis) are categories

    """
    skew = (a - b)/(a + b)
    return skew

def error_bar_array(df, interval=0.95):
    data = ci(df, ci=interval)
    return np.vstack([data['lb_dist'], data['ub_dist']])

