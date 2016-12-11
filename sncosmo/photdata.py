# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Convenience functions for photometric data."""
from __future__ import division

from collections import OrderedDict
import copy
import math

import numpy as np
from astropy.table import Table

from .utils import alias_map
from .bandpasses import get_bandpass
from .magsystems import get_magsystem

# deprecated (private!) functions: make them available where they used to be
from ._deprecated import standardize_data, normalize_data


PHOTDATA_ALIASES = OrderedDict([
    ('time', set(['time', 'date', 'jd', 'mjd', 'mjdobs', 'mjd_obs'])),
    ('band', set(['band', 'bandpass', 'filter', 'flt'])),
    ('flux', set(['flux', 'f'])),
    ('fluxerr', set(['fluxerr', 'fe', 'fluxerror', 'flux_error', 'flux_err'])),
    ('zp', set(['zp', 'zpt', 'zeropoint', 'zero_point'])),
    ('zpsys', set(['zpsys', 'zpmagsys', 'magsys']))
    ])


class PhotometricData(object):
    """Internal standardized representation of photometric data table.

    Has attributes ``time``, ``band``, ``flux``, ``fluxerr``, ``zp`` and
    ``zpsys``, which are all numpy arrays of the same length sorted by
    ``time``. This is intended for use within sncosmo; its implementation
    may change without warning in future versions.

    Parameters
    ----------
    data : `~astropy.table.Table`, dict, `~numpy.ndarray`
        Astropy Table, dictionary of arrays or structured numpy array
        containing the "correct" column names.
    """

    def __init__(self, data):
        # get column names in input data
        if isinstance(data, Table):
            colnames = data.colnames
        elif isinstance(data, np.ndarray):
            colnames = data.dtype.names
        elif isinstance(data, dict):
            colnames = data.keys()
        else:
            raise ValueError('unrecognized data type')

        mapping = alias_map(colnames, PHOTDATA_ALIASES)

        self.time = np.asarray(data[mapping['time']])
        self.band = np.asarray(data[mapping['band']])
        self.flux = np.asarray(data[mapping['flux']])
        self.fluxerr = np.asarray(data[mapping['fluxerr']])
        self.zp = np.asarray(data[mapping['zp']])
        self.zpsys = np.asarray(data[mapping['zpsys']])

        # ensure columns are equal length
        if isinstance(data, dict):
            if not (len(self.time) == len(self.band) == len(self.flux) ==
                    len(self.fluxerr) == len(self.zp) == len(self.zpsys)):
                raise ValueError("unequal column lengths")

        # Sort by time, if necessary.
        if not np.all(np.ediff1d(self.time) >= 0.0):
            idx = np.argsort(self.time)
            self.time = self.time[idx]
            self.band = self.band[idx]
            self.flux = self.flux[idx]
            self.fluxerr = self.fluxerr[idx]
            self.zp = self.zp[idx]
            self.zpsys = self.zpsys[idx]

    def __len__(self):
        return len(self.time)

    def __getitem__(self, key):
        newdata = copy.copy(self)  # avoid going through __init__
        newdata.time = self.time[key]
        newdata.band = self.band[key]
        newdata.flux = self.flux[key]
        newdata.fluxerr = self.fluxerr[key]
        newdata.zp = self.zp[key]
        newdata.zpsys = self.zpsys[key]
        return newdata

    def normalized(self, zp=25., zpsys='ab'):
        """Return a copy of the data with all flux and fluxerr values
        normalized to the given zeropoint.
        """

        normmagsys = get_magsystem(zpsys)
        factor = np.empty(len(self), dtype=np.float)

        for b in set(self.band.tolist()):
            idx = self.band == b
            b = get_bandpass(b)

            bandfactor = 10.**(0.4 * (zp - self.zp[idx]))
            bandzpsys = self.zpsys[idx]
            for ms in set(bandzpsys):
                idx2 = bandzpsys == ms
                ms = get_magsystem(ms)
                bandfactor[idx2] *= (ms.zpbandflux(b) /
                                     normmagsys.zpbandflux(b))
            factor[idx] = bandfactor

        newdata = copy.copy(self)
        newdata.flux = factor * self.flux
        newdata.fluxerr = factor * self.fluxerr
        newdata.zp = np.full(len(self), zp, dtype=np.float64)
        newdata.zpsys = np.full(len(self), zpsys, dtype=np.array(zpsys).dtype)

        return newdata


def photometric_data(data):
    if isinstance(data, PhotometricData):
        return data
    else:
        return PhotometricData(data)


# Generate docstring: table of aliases

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

lines = [
    '',
    '  '.join([10 * '=', 60 * '=', 50 * '=', 50 * '=']),
    '{0:10}  {1:60}  {2:50}  {3:50}'
    .format('Column', 'Acceptable aliases (case-independent)',
            'Description', 'Type')
    ]
lines.append(lines[1])
for colname in PHOTDATA_ALIASES:
    alias_list = ', '.join([repr(a) for a in PHOTDATA_ALIASES[colname]])
    line = '{0:10}  {1:60}  {2:50}  {3:50}'.format(
        colname,
        alias_list,
        _photdata_descriptions[colname],
        _photdata_types[colname])
    lines.append(line)
lines.extend([lines[1], ''])
__doc__ = '\n'.join(lines)
