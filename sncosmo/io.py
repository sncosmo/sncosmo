# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

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
        f = open(name_or_obj, 'rb')
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
