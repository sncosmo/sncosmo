from sncosmo.photdata import PHOTDATA_ALIASES

# Generate docstring: table of aliases

# Descriptions for docstring only.
_photdata_descriptions = {
    'time': 'Time of observation in days',
    'band': 'Bandpass of observation',
    'flux': 'Flux of observation',
    'fluxerr': 'Gaussian uncertainty on flux',
    'zp': 'Zeropoint corresponding to flux',
    'zpsys': 'Magnitude system for zeropoint',
    'fluxcov': 'Covariance between observations (array; optional)'
    }

_photdata_types = {
    'time': 'float',
    'band': 'str',
    'flux': 'float',
    'fluxerr': 'float',
    'zp': 'float',
    'zpsys': 'str',
    'fluxcov': 'ndarray'
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
