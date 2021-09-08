# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Convenience functions for photometric data."""

import copy
from collections import OrderedDict

import numpy as np
from astropy.table import Table

from .bandpasses import get_bandpass
from .magsystems import get_magsystem
from .utils import alias_map

__all__ = ['select_data']

PHOTDATA_ALIASES = OrderedDict([
    ('time', {'time', 'date', 'jd', 'mjd', 'mjdobs', 'mjd_obs'}),
    ('band', {'band', 'bandpass', 'filter', 'flt'}),
    ('flux', {'flux', 'f'}),
    ('fluxerr', {'fluxerr', 'fe', 'fluxerror', 'flux_error', 'flux_err'}),
    ('zp', {'zp', 'zpt', 'zeropoint', 'zero_point'}),
    ('zpsys', {'zpsys', 'zpmagsys', 'magsys'}),
    ('fluxcov', {'cov', 'covar', 'covariance', 'covmat', 'fluxcov'})
    ])

PHOTDATA_REQUIRED_ALIASES = ('time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys')


class PhotometricData(object):
    """Internal standardized representation of photometric data table.

    Has attributes ``time``, ``band``, ``flux``, ``fluxerr``, ``zp``
    and ``zpsys``, which are all numpy arrays of the same length
    sorted by ``time``. ``band`` is an array of Bandpass objects. This
    is intended for use within sncosmo; its implementation may change
    without warning in future versions.

    Has attribute ``fluxcov`` which may be ``None``.

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

        mapping = alias_map(colnames, PHOTDATA_ALIASES,
                            required=PHOTDATA_REQUIRED_ALIASES)

        self.time = np.asarray(data[mapping['time']])

        # ensure self.band contains Bandpass objects. (We could check
        # if the original array already contains all bandpass objects,
        # but constructing a new array is simpler.)
        band_orig = data[mapping['band']]
        self.band = np.empty(len(band_orig), dtype=object)
        for i in range(len(band_orig)):
            self.band[i] = get_bandpass(band_orig[i])

        self.flux = np.asarray(data[mapping['flux']])
        self.fluxerr = np.asarray(data[mapping['fluxerr']])
        self.zp = np.asarray(data[mapping['zp']])
        self.zpsys = np.asarray(data[mapping['zpsys']])
        self.fluxcov = (np.asarray(data[mapping['fluxcov']])
                        if 'fluxcov' in mapping else None)

        # ensure columns are equal length
        if isinstance(data, dict):
            if not (len(self.time) == len(self.band) == len(self.flux) ==
                    len(self.fluxerr) == len(self.zp) == len(self.zpsys)):
                raise ValueError("unequal column lengths")

        # handle covariance if present
        if self.fluxcov is not None:
            # check shape OK
            n = len(self.time)
            if self.fluxcov.shape != (n, n):
                raise ValueError(
                    "Flux covariance must be shape (N, N). Did you slice "
                    "the data? Use ``sncosmo.select_data(data, mask)`` in "
                    "place of ``data[mask]`` to properly slice covariance.")

    def sort_by_time(self):
        if not np.all(np.ediff1d(self.time) >= 0.0):
            idx = np.argsort(self.time)
            self.time = self.time[idx]
            self.band = self.band[idx]
            self.flux = self.flux[idx]
            self.fluxerr = self.fluxerr[idx]
            self.zp = self.zp[idx]
            self.zpsys = self.zpsys[idx]
            self.fluxcov = (None if self.fluxcov is None else
                            self.fluxcov[np.ix_(idx, idx)])

    def __len__(self):
        return len(self.time)

    def __getitem__(self, key):
        newdata = copy.copy(self)
        newdata.time = self.time[key]
        newdata.band = self.band[key]
        newdata.flux = self.flux[key]
        newdata.fluxerr = self.fluxerr[key]
        newdata.zp = self.zp[key]
        newdata.zpsys = self.zpsys[key]
        newdata.fluxcov = (None if self.fluxcov is None else
                           self.fluxcov[np.ix_(key, key)])

        return newdata

    def normalized(self, zp=25., zpsys='ab'):
        """Return a copy of the data with all flux and fluxerr values
        normalized to the given zeropoint.
        """

        factor = self._normalization_factor(zp, zpsys)

        newdata = copy.copy(self)
        newdata.flux = factor * self.flux
        newdata.fluxerr = factor * self.fluxerr
        newdata.zp = np.full(len(self), zp, dtype=np.float64)
        newdata.zpsys = np.full(len(self), zpsys, dtype=np.array(zpsys).dtype)
        if newdata.fluxcov is not None:
            newdata.fluxcov = factor * factor[:, None] * self.fluxcov

        return newdata

    def normalized_flux(self, zp=25., zpsys='ab'):
        return self._normalization_factor(zp, zpsys) * self.flux

    def _normalization_factor(self, zp, zpsys):
        """Factor such that multiplying by this amount brings all fluxes onto
        the given zeropoint and zeropoint system."""

        normmagsys = get_magsystem(zpsys)
        factor = np.empty(len(self), dtype=float)

        for b in set(self.band.tolist()):
            mask = self.band == b
            bandfactor = 10.**(0.4 * (zp - self.zp[mask]))
            bandzpsys = self.zpsys[mask]
            for ms in set(bandzpsys):
                mask2 = bandzpsys == ms
                ms = get_magsystem(ms)
                bandfactor[mask2] *= (ms.zpbandflux(b) /
                                      normmagsys.zpbandflux(b))
            factor[mask] = bandfactor

        return factor


def photometric_data(data):
    if isinstance(data, PhotometricData):
        return data
    else:
        return PhotometricData(data)


def select_data(data, index):
    """Convenience function for indexing photometric data with covariance.

    This is like ``data[index]`` on an astropy Table, but handles
    covariance columns correctly.

    Parameters
    ----------
    data : `~astropy.table.Table`
        Table of photometric data.
    index : slice or array or int
        Row selection to apply to table.

    Returns
    -------
    `~astropy.table.Table`

    Examples
    --------

    We have a small table of photometry with a covariance column and we
    want to select some rows based on a mask:

    >>> data = Table([[1., 2., 3.],
    ...               ['a', 'b', 'c'],
    ...               [[1.1, 1.2, 1.3],
    ...                [2.1, 2.2, 2.3],
    ...                [3.1, 3.2, 3.3]]],
    ...               names=['time', 'x', 'cov'])
    >>> mask = np.array([True, True, False])

    Selecting directly on the table, the covariance column is not sliced
    in each row: it has shape (2, 3) when it should be (2, 2):

    >>> data[mask]
    <Table length=2>
      time   x    cov [3]
    float64 str1  float64
    ------- ---- ----------
        1.0    a 1.1 .. 1.3
        2.0    b 2.1 .. 2.3

    Using ``select_data`` solves this:

    >>> sncosmo.select_data(data, mask)
    <Table length=2>
      time   x    cov [2]
    float64 str1  float64
    ------- ---- ----------
        1.0    a 1.1 .. 1.2
        2.0    b 2.1 .. 2.2

    """
    mapping = alias_map(data.colnames,
                        {'fluxcov': PHOTDATA_ALIASES['fluxcov']})
    result = data[index]
    if 'fluxcov' in mapping:
        colname = mapping['fluxcov']
        fluxcov = result[colname][:, index]

        # replace_column method not available in astropy 1.0
        i = result.index_column(colname)
        del result[colname]
        result.add_column(fluxcov, i)

    return result
