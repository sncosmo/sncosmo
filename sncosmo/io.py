from collections import OrderedDict
import numpy as np
from . import utils

__all__ = ['writelc', 'readlc']

def _stripcomment(line, char='#'):
    pos = line.find(char)
    if pos == -1: return line
    else: return line[:pos]

def read_griddata(name_or_obj):
    """Read 2-d grid data from a text file.

    Each line has values `x0 x1 y`. Space separated.
    x1 values are only read for first x0 value. Others are assumed
    to match.

    Parameters
    ----------
    filename : str or file-like object

    Returns
    -------
    x0 : numpy.ndarray
        1-d array.
    x1 : numpy.ndarray
        1-d array.
    y : numpy.ndarray
        2-d array of shape (len(x0), len(x1)).
    """

    if isinstance(name_or_obj, basestring):
        f = open(filename, 'rb')
    else:
        f = name_or_obj

    x0 = []    # x0 values.
    x1 = None  # x1 values for first x0 value, assume others are the same.
    y = []     # 2-d array of internal values

    x0_current = None
    x1_current = []
    y1_current = []
    for line in f:
        stripped_line = _stripcomment(line)
        if len(stripped_line) == 0: continue
        x0_tmp, x1_tmp, y_tmp = map(float, stripped_line.split())
        if x0_current is None: x0_current = x0_tmp  #Initialize first time

        # If there is a new x0 value, ingest the old one and reset values
        if x0_tmp != x0_current:
            x0.append(x0_current)
            if x1 is None: x1 = x1_current
            y.append(y1_current)

            x0_current = x0_tmp
            x1_current = []
            y1_current = []

        x1_current.append(x1_tmp)
        y1_current.append(y_tmp)

    # Ingest the last x0 value and y1 array
    x0.append(x0_current)
    y.append(y1_current)

    f.close()
    return np.array(x0), np.array(x1), np.array(y)


def writelc(data, fname, meta=None, fmt=None):
    """Write light curve data."""
    
    f = open(fname, 'w')
    f.write('#time band flux fluxerr zp zpsys\n')
    for i in range(len(data['time'])):
        f.write('{:f} {:s} {:f} {:f} {:f} {:s}\n'.format(
                data['time'][i], data['band'][i], data['flux'][i],
                data['fluxerr'][i], data['zp'][i], data['zpsys'][i]))
    f.close()

def readlc(fname, fmt=None):
    """Read light curve data.

    Returns
    -------
    meta : dict
    data : ndarray
    """

    datarows = []
    f = open(fname, 'r')
    for line in f.readlines():
        line = _stripcomment(line)
        if len(line) == 0: continue
        row = []
        for item in line.split():
            try:
                item = int(item)
            except:
                try:
                    item = float(item)
                except:
                    pass
            row.append(item)
        datarows.append(row)
    return utils.rows_to_array(datarows, ['time', 'band', 'flux', 'fluxerr',
                                          'zp', 'zpsys'])
