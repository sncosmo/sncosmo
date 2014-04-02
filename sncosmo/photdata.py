# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Convenience functions for photometric data."""
from __future__ import division

import math
import numpy as np
from astropy.utils import OrderedDict as odict
from astropy.table import Table
from .spectral import get_magsystem, get_bandpass

_photdata_aliases = odict([
    ('time', set(['time', 'date', 'jd', 'mjd', 'mjdobs'])),
    ('band', set(['band', 'bandpass', 'filter', 'flt'])),
    ('flux', set(['flux', 'f'])),
    ('fluxerr', set(['fluxerr', 'fe', 'fluxerror', 'flux_error', 'flux_err'])),
    ('zp', set(['zp', 'zpt', 'zeropoint', 'zero_point'])),
    ('zpsys', set(['zpsys', 'zpmagsys', 'magsys']))
    ])
    
# Descriptions for docstring only.
_photdata_descriptions = {
    'time': 'Time of observation in days',
    'band': 'Bandpass of observation',
    'flux': 'Flux of observation',
    'fluxerr': 'Gaussian uncertainty on flux',
    'zp': 'Zeropoint corresponding to flux',
    'zpsys': 'Magnitude system for zeropoint'
    }
_photdata_types = {
    'time': 'float',
    'band': 'str',
    'flux': 'float',
    'fluxerr': 'float',
    'zp': 'float',
    'zpsys': 'str'
    }

def dict_to_array(d):
    """Convert a dictionary of lists (or single values) to a structured
    numpy.ndarray."""

    # Convert all lists/values to 1-d arrays, in order to let numpy
    # figure out the necessary size of the string arrays.
    new_d = odict()
    for key in d: 
        new_d[key] = np.atleast_1d(d[key])

    # Determine dtype of output array.
    dtype = [(key, arr.dtype) for key, arr in new_d.iteritems()]

    # Initialize ndarray and then fill it.
    col_len = max([len(v) for v in new_d.values()])
    result = np.empty(col_len, dtype=dtype)
    for key in new_d:
        result[key] = new_d[key]

    return result

def standardize_data(data):
    """Standardize photometric data by converting to a structured numpy array
    with standard column names (if necessary) and sorting entries in order of
    increasing time.

    Parameters
    ----------
    data : `~astropy.table.Table`, `~numpy.ndarray` or dict

    Returns
    -------
    standardized_data : `~numpy.ndarray`
    """
    if isinstance(data, Table):
        data = np.asarray(data)

    if isinstance(data, np.ndarray):
        colnames = data.dtype.names

        # Check if the data already complies with what we want
        # (correct column names & ordered by date)
        if (set(colnames) == set(_photdata_aliases.keys()) and
            np.all(np.ediff1d(data['time']) >= 0.)):
            return data

    elif isinstance(data, dict):
        colnames = data.keys()

    else:
        raise ValueError('Unrecognized data type')

    # Create mapping from lowercased column names to originals
    lower_to_orig = dict([(colname.lower(), colname) for colname in colnames])
        
    # Set of lowercase column names
    lower_colnames = set(lower_to_orig.keys())

    orig_colnames_to_use = []
    for aliases in _photdata_aliases.values():
        i = lower_colnames & aliases
        if len(i) != 1:
            raise ValueError('Data must include exactly one column from {0} '
                             '(case independent)'.format(', '.join(aliases)))
        orig_colnames_to_use.append(lower_to_orig[i.pop()])

    if isinstance(data, np.ndarray):
        new_data = data[orig_colnames_to_use].copy()
        new_data.dtype.names = _photdata_aliases.keys()

    else:
        new_data = odict()
        for newkey, oldkey in zip(_photdata_aliases.keys(),
                                  orig_colnames_to_use):
            new_data[newkey] = data[oldkey]
        
        new_data = dict_to_array(new_data)

    # Sort by time, if necessary.
    if not np.all(np.ediff1d(new_data['time']) >= 0.):
        new_data.sort(order=['time'])

    return new_data

def normalize_data(data, zp=25., zpsys='ab'):
    """Return a copy of the data with all flux and fluxerr values normalized
    to the given zeropoint. Assumes data has already been standardized.

    Parameters
    ----------
    data : `~numpy.ndarray`
        Structured array.
    zp : float
    zpsys : str

    Returns
    -------
    normalized_data : `~numpy.ndarray`
    """

    normmagsys = get_magsystem(zpsys)
    factor = np.empty(len(data), dtype=np.float)
    
    for b in set(data['band'].tolist()):
        idx = data['band'] == b
        b = get_bandpass(b)

        bandfactor = 10.**(0.4 * (zp - data['zp'][idx]))
        bandzpsys = data['zpsys'][idx]
        for ms in set(bandzpsys):
            idx2 = bandzpsys == ms
            ms = get_magsystem(ms)
            bandfactor[idx2] *= (ms.zpbandflux(b) / normmagsys.zpbandflux(b))
        
        factor[idx] = bandfactor

    normalized_data = odict([('time', data['time']),
                             ('band', data['band']),
                             ('flux', data['flux'] * factor),
                             ('fluxerr', data['fluxerr'] * factor),
                             ('zp', zp),
                             ('zpsys', zpsys)])
    return dict_to_array(normalized_data)

# Generate docstring: table of aliases
lines = [
    '',
    '  '.join([60 * '=', 50 * '=', 50 * '=']),
    '{0:60}  {1:50}  {2:50}'
    .format('Acceptable column names (case-independent)',
            'Description', 'Type')
    ]
lines.append(lines[1])

for field, aliases in _photdata_aliases.iteritems():
    lines.append('{0:60}  {1:50}  {2:50}'
                 .format(', '.join([repr(a) for a in aliases]),
                         _photdata_descriptions[field],
                         _photdata_types[field]))

lines.extend([lines[1], ''])
__doc__ = '\n'.join(lines)
