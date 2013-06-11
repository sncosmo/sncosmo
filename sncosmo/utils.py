# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""General utilities"""
import numpy as np

from astropy.utils import OrderedDict

__all__ = ['dict_to_array', 'transpose', 'rows_to_dict', 'rows_to_array']

def dict_to_array(d):
    """Convert a dictionary of lists (of equal length) to a structured
    numpy.ndarray"""

    # first convert all lists to 1-d arrays, in order to let numpy
    # figure out the necessary size of the string arrays.
    for key in d: 
        d[key] = np.array(d[key])

    # Determine dtype of output array.
    dtypelist = []
    for key in d:
        dtypelist.append((key, d[key].dtype))
    
    # Initialize ndarray and then fill it.
    firstkey = d.keys()[0]
    col_len = len(d[firstkey])
    result = np.empty(col_len, dtype=dtypelist)
    for key in d:
        result[key] = d[key]

    return result


def transpose(it):
    """transpose a 2-d iterable (list of lists or list of tuples)."""

    if len(it) == 0:
        return []
    rowlen = len(it[0])
    if any([len(row) != rowlen for row in it]):
        raise ValueError("Iterable must have all rows equal length (first row has length {:d})".format(rowlen))
    return [[row[i] for row in it] for i in range(rowlen)]


def rows_to_dict(rows, colnames):
    """Convert a 2-d iterable (e.g. list of lists) where each element is a row
    to an OrderedDict where each element is a column."""

    cols = transpose(rows)
    if len(cols) != len(colnames):
        raise ValueError('length of each row must match length of colnames')
    d = OrderedDict()
    for i in range(len(cols)):
        d[colnames[i]] = cols[i]
    return d

def rows_to_array(rows, colnames):
    """Convert a 2-d iterable (e.g. list of lists) to a structured ndarray."""
    return dict_to_array(rows_to_dict(rows, colnames))
