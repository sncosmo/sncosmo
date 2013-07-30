# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Convenience class for photometric data."""
from __future__ import division

import math
import numpy as np
from astropy.utils import OrderedDict
from astropy.table import Table
from .spectral import get_magsystem

_photdata_aliases = OrderedDict([
    ('time', set(['time', 'date', 'jd', 'mjd', 'mjdobs'])),
    ('band', set(['band', 'bandpass', 'filter', 'flt'])),
    ('flux', set(['flux', 'f'])),
    ('fluxerr', set(['fluxerr', 'fe', 'fluxerror', 'flux_error', 'flux_err'])),
    ('zp', set(['zp', 'zpt', 'zeropoint', 'zero_point'])),
    ('zpsys', set(['zpsys', 'zpmagsys', 'magsys']))
    ])
    
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
    'band': 'str or `~sncosmo.Bandpass` instance',
    'flux': 'float',
    'fluxerr': 'float',
    'zp': 'float',
    'zpsys': 'str or `~sncosmo.spectral.MagSystem` instance'
    }

# TODO: hold normalized flux internally? (or just convert all flux values?)
# TODO: indexing to extract subsets of the data (e.g., just "good" bands)
# TODO: run get_bandpass and get_magsystem at the start? How will this affect
#       np.unique (used in model.bandflux and many other places)?
class PhotData(object):
    """Class for holding photometric data. This class is intended for internal
    use only."""

    def __init__(self, data):

        # For now, use the Table class internally to parse the data.
        # In the future we may want to subclass Table instead.
        t = Table(data, copy=False)
        data = np.array(t, copy=False)
        colnames = set([name.lower() for name in t.colnames])

        for attribute, aliases in _photdata_aliases.iteritems():
            i = colnames & aliases
            if len(i) != 1:
                raise ValueError(
                    'Data must include exactly one column from: ' +
                    ', '.join(list(aliases)) + ' (case-independent)'
                    )
            self.__dict__[attribute] = data[i.pop()]

        self.length = len(data)

    def __len__(self):
        return self.length

    def normalized_flux(self, zp=25., zpsys='ab', include_err=False):
        """Return flux values normalized to a common zeropoint and magnitude
        system."""

        magsys = get_magsystem(magsys)
        factor = np.empty(self.length, dtype=np.float)

        for i in range(self.length):
            ms = get_magsystem(self.zpsys[i])
            factor[i] = (ms.zpbandflux(self.band[i]) /
                         magsys.zpbandflux(self.band[i]) *
                         10.**(0.4 * (zp - self.zp[i])))

        if include_err:
            return self.flux * factor, self.fluxerr * factor
        return self.flux * factor

# Generate docstring: table of aliases
lines = [
    '',
    '  '.join([60 * '=', 50 * '=', 50 * '=']),
    '{:60}  {:50}  {:50}'.format('Acceptable column names (case-independent)', 'Description', 'Type')
    ]
lines.append(lines[1])

for field, aliases in _photdata_aliases.iteritems():
    lines.append('{:60}  {:50}  {:50}'
                 .format(', '.join([repr(a) for a in aliases]),
                         _photdata_descriptions[field],
                         _photdata_types[field]))

lines.extend([lines[1], ''])
__doc__ = '\n'.join(lines)
