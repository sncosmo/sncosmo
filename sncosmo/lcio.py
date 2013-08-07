# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for supernova light curve I/O"""

# This module is designed to be self-contained, with only a dependency on
# numpy. Since OrderedDict is only in 2.7+, try to get `OrderedDict` from
# collections, then from astropy, then just use a regular `dict`.
from warnings import warn
import json
import sys
import re
import numpy as np
try:
    from collections import OrderedDict as odict
except ImportError:
    try:
        from astropy.utils import OrderedDict as odict
    except ImportError:
        odict = dict

__all__ = ['read_lc', 'write_lc']

# Helper functions ========================================================== #
def _cast_str(s):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s.strip()

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

# Reader: csv =============================================================== #
def _read_csv(f, **kwargs):

    delim = kwargs.get('delim', None)
    metachar = kwargs.get('metachar', '@')
    commentchar = kwargs.get('commentchar', '#')

    meta = odict()
    colnames = []
    cols = []
    readingdata = False
    for line in f:
        
        # strip leading & trailing whitespace, newline, and comments
        line = line.strip()
        pos = line.find(commentchar)
        if pos > -1:
            line = line[:pos]
        if len(line) == 0: continue

        if not readingdata:
            # Read metadata
            if line[0] == metachar:
                pos = line.find(' ')  # Find first space.
                if pos in [-1, 1]:  # Space must exist and key must exist.
                    raise ValueError('Incorrectly formatted metadata line: '+
                                     line)
                meta[line[1:pos]] = _cast_str(line[pos:])
                continue

            # Read header line
            for item in line.split(delim):
                colnames.append(item.strip())
                cols.append([])
            readingdata = True
            continue

        # Now we're reading data
        items = line.split(delim)
        for col, item in zip(cols, items):
            col.append(_cast_str(item))

    data = odict(zip(colnames, cols))
    return meta, data

# Reader: salt2 ============================================================= #

# SALT2 conversions upon reading:
# Names are converted to lowercase, then the following lookup table is used.
SALT2KEY_TO_KEY = {'redshift': 'z',
                   'z_heliocentric': 'z_helio',
                   'filter': 'band'}
def _salt2_rename_keys(d):
    newd = odict()
    for key, val in d.iteritems():
        key = key.lower()
        if key in SALT2KEY_TO_KEY:
            key = SALT2KEY_TO_KEY[key]
        newd[key] = val
    return newd

def _read_salt2(f, **kwargs):
    """Read a new-style SALT2 file.
    
    Such a file has metadata on lines starting with '@' and column names
    on lines starting with '#' and containing a ':' after the column name.
    There is optionally a line containing '#end' before the start of data.
    """

    raw = kwargs.get('raw', False)

    meta = odict()
    colnames = []
    cols = []
    readingdata = False
    for line in f:
        
        # strip leading & trailing whitespace & newline
        line = line.strip()
        if len(line) == 0: continue

        if not readingdata:
            # Read metadata
            if line[0] == '@':
                pos = line.find(' ')  # Find first space.
                if pos in [-1, 1]:  # Space must exist and key must exist.
                    raise ValueError('Incorrectly formatted metadata line: '+
                                     line)
                meta[line[1:pos]] = _cast_str(line[pos:])
                continue

            # Read header line
            if line[0] == '#':
                pos = line.find(':')
                if pos in [-1, 1]: continue  # comment line
                colname = line[1:pos].strip()
                if colname == 'end': continue 
                colnames.append(colname)
                cols.append([])
                continue

            # If the first non-whitespace character is not '@' or '#',
            # assume the line is the first data line.
            readingdata = True

        # strip comments
        pos = line.find('#')
        if pos > -1:
            line = line[:pos]
        if len(line) == 0: continue

        # Now we're reading data
        items = line.split()
        for col, item in zip(cols, items):
            col.append(_cast_str(item))

    data = odict(zip(colnames, cols))

    # convert keys, if requested
    if not raw:
        meta = _salt2_rename_keys(meta)
        data = _salt2_rename_keys(data)

    return meta, data

# Reader: json ============================================================== #
def _read_json(f, **kwargs):
    t = json.load(f, encoding=sys.getdefaultencoding())
    d = {}
    for key, value in t['data'].items():
        d[key.encode('ascii')] = value
    return t['meta'], d


# Reader: fits ============================================================== #

def _read_fits(f, **kwargs):
    try:
        from astropy.io import fits
    except ImportError:
        try:
            import pyfits as fits
        except ImportError:
            raise ValueError("Either pyfits or astropy is required for 'fits'"
                             " format")
    hdulist = fits.open(f) 

    # Parse all table objects
    tables = odict()
    for ihdu, hdu_item in enumerate(hdulist):
        if isinstance(hdu_item, (fits.TableHDU, fits.BinTableHDU,
                                 fits.GroupsHDU)):
            tables[ihdu] = hdu_item

    if len(tables) > 1:
        hdu = tables.keys()[0]
        warn("hdu= was not specified but multiple tables"
             " are present, reading in first available"
             " table (hdu={0})".format(hdu))
        hdu = hdulist.index_of(hdu)
        if hdu in tables:
            table = tables[hdu]
        else:
            raise ValueError("No table found in hdu={0}".format(hdu))

    elif len(tables) == 1:
        table = tables[tables.keys()[0]]
    else:
        raise ValueError("No table found")

    # Create a metadata dictionary.
    meta = odict()
    for key, value, comment in table.header.cards:
        if key in ['COMMENT', 'HISTORY']:
            if key in meta:
                meta[key].append(value)
            else:
                meta[key] = [value]

        elif key in meta:  # key is duplicate
            if isinstance(meta[key], list):
                meta[key].append(value)
            else:
                meta[key] = [meta[key], value]

        elif (is_column_keyword(key.upper()) or
              key.upper() in REMOVE_KEYWORDS):
            pass

        else:
            meta[key] = value

    data = table.data.view(np.ndarray)
    hdulist.close()

    return meta, data

# All readers =============================================================== #
READERS = {'csv': _read_csv,
           'json': _read_json,
           'fits': _read_fits,
           'salt2': _read_salt2}
def read_lc(fname, fmt='csv', array=True, **kwargs):
    """Read light curve data.

    Parameters
    ----------
    fname : str
        Filename 
    fmt : {'csv', 'salt2', 'json'}, optional
        Format of file. Default is 'csv'. 'salt2' is the new format available
        in snfit version >= 2.3.0.
    array : bool, optional
        If True (default), returned data is a numpy array, Otherwise, data
        is a dictionary of lists (representing columns).
    delim : str, optional
        **[csv only]** Used to split entries on a line. Default is `None`.
        Extra whitespace is ignored.
    metachar : str, optional
        **[csv only]** Lines whose first non-whitespace character is `metachar`
        are treated as metadata lines, where the key and value are split on
        the first whitespace. Default is '@'
    commentchar : str, optional
        **[csv only]** One-character string indicating a comment. Default is
        '#'.
    raw : bool, optional
        **[salt2 only]** By default, the SALT2 reader converts all column names
        and metadata keys to lowercase and renames a few of them. Set to True
        to override this. Default is False.

    Returns
    -------
    meta : dict
        A (possibly empty) dictionary of metadata in the file.
    data : `~numpy.ndarray` or dict
        Data.
    """

    if fmt not in READERS:
        raise ValueError("Reader not defined for format '{}'. Options: "
                         .format(fmt) + ", ".join(READERS.keys()))

    with open(fname, 'rb') as f:
        meta, data = READERS[fmt](f, **kwargs)

    if array and not isinstance(data, np.ndarray):
        data = dict_to_array(data)

    return meta, data

# =========================================================================== #
# Writers                                                                     #
# =========================================================================== #

# Writer: csv =============================================================== #
def _write_csv(f, data, meta, **kwargs):
    
    delim = kwargs.get('delim', ' ')
    metachar = kwargs.get('metachar', '@')
    
    if meta is not None:
        for key, val in meta.iteritems():
            f.write('{}{}{}{}\n'.format(metachar, key, delim, str(val)))

    keys = data.dtype.names
    length = len(data)
    
    f.write(delim.join(keys))
    f.write('\n')
    for i in range(length):
        f.write(delim.join([str(data[key][i]) for key in keys]))
        f.write('\n')

# Writer: salt2 ============================================================= #
KEY_TO_SALT2KEY_META = {
    'Z': 'REDSHIFT',              # Not sure if this is used.
    'Z_HELIOCENTRIC': 'Z_HELIO',
    'MAGSYS': 'MagSys',
    'Z_SOURCE': 'z_source'}
KEY_TO_SALT2KEY_COLUMN = {
    'Mjd': 'Date',
    'Time': 'Date',
    'Flux': 'FluxPsf',
    'Fluxpsf': 'FluxPsf',
    'Fluxerr': 'FluxPsferr',
    'Fluxpsferr': 'FluxPsferr',
    'Airmass': 'AirMass',
    'Zp': 'ZP',
    'Zpsys': 'MagSys',
    'Magsys': 'MagSys',
    'Band': 'Filter'}

def _write_salt2(f, data, meta, **kwargs):

    raw = kwargs.get('raw', False)
    pedantic = kwargs.get('pedantic', True)
    
    if meta is not None:
        for key, val in meta.iteritems():
            if not raw:
                key = key.upper()
                key = KEY_TO_SALT2KEY_META.get(key, key)
            f.write('@{} {}\n'.format(key, str(val)))

    keys = data.dtype.names
    length = len(data)

    # Write column names
    keys_as_written = []
    for key in keys:
        if not raw:
            key = key.capitalize()
            key = KEY_TO_SALT2KEY_COLUMN.get(key, key)
        f.write('#{} :\n'.format(key))
        keys_as_written.append(key)
    f.write('#end :\n')

    # Check that necessary fields exist
    if pedantic:
        if not ('Filter' in keys_as_written and 'MagSys' in keys_as_written):
            raise ValueError('photometry data missing required some fields '
                             ': Filter, MagSys')

    # Write the data itself
    for i in range(length):
        f.write(' '.join([str(data[key][i]) for key in keys]))
        f.write('\n')

# Writer: snana ============================================================= #
KEY_TO_SNANAKEY_COLUMN = {
    'TIME': 'MJD',
    'DATE': 'MJD',
    'FILTER': 'FLT',
    'BAND': 'FLT',
    'FLUX': 'FLUXCAL',
    'FLUXERR': 'FLUXCALERR',
    'ZP': 'ZPT',
    'ZEROPOINT': 'ZPT'
    }
KEY_TO_SNANAKEY_META = {
    'DEC': 'DECL'
    }
SNANA_REQUIRED_META = ['RA', 'DECL', 'SURVEY', 'FILTERS', 'MWEBV']
SNANA_REQUIRED_COLUMN = ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR', 'ZPT']
def _write_snana(f, data, meta, **kwargs):

    raw = kwargs.get('raw', False)
    pedantic = kwargs.get('pedantic', True)

    # Write metadata
    keys_as_written = []
    if meta is not None:
        for key, val in meta.iteritems():
            if not raw:
                key = key.upper()
                key = KEY_TO_SNANAKEY_META.get(key, key)
            f.write('{}: {}\n'.format(key, str(val)))
            keys_as_written.append(key)

    # Check that necessary metadata was written
    if pedantic:
        for key in SNANA_REQUIRED_META:
            if key not in keys_as_written:
                raise ValueError('Missing required metadata kw: ' + key)
    
    # Get column names and data length
    keys = data.dtype.names
    length = len(data)
    
    # Convert column names
    keys_to_write = []
    for key in keys:
        if not raw:
            key = key.upper()
            key = KEY_TO_SNANAKEY_COLUMN.get(key, key)
        keys_to_write.append(key)

    # Check that necessary column names are included
    if pedantic:
        for key in SNANA_REQUIRED_COLUMN:
            if key not in keys_to_write:
                raise ValueError('Missing required column name: ' + key)

    # Write the header
    f.write('\n'
            '# ==========================================\n'
            '# TERSE LIGHT CURVE OUTPUT:\n'
            '#\n'
            'NOBS: {:d}\n'
            'NVAR: {:d}\n'
            'VARLIST: {}\n'
            .format(length, len(keys), ' '.join(keys_to_write)))

    # Write data
    for i in range(length):
        f.write('OBS: ')
        f.write(' '.join([str(data[key][i]) for key in keys]))
        f.write('\n')

# Writer: json ============================================================== #
def _write_json(f, data, meta, **kwargs):

    # Build a dictionary of pure-python objects
    output = odict([('meta', meta),
                    ('data', odict())])
    for key in data.dtype.names:
        output['data'][key] = data[key].tolist()
    json.dump(output, f, encoding=sys.getdefaultencoding())
    del output

# Writer: fits ============================================================== #

# Adapted from astropy.io.fits.connect.write_table_fits

# Keywords to remove for all tables that are read in
REMOVE_KEYWORDS = ['XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                   'PCOUNT', 'GCOUNT', 'TFIELDS']

# Column-specific keywords
COLUMN_KEYWORDS = ['TFORM[0-9]+',
                   'TBCOL[0-9]+',
                   'TSCAL[0-9]+',
                   'TZERO[0-9]+',
                   'TNULL[0-9]+',
                   'TTYPE[0-9]+',
                   'TUNIT[0-9]+',
                   'TDISP[0-9]+',
                   'TDIM[0-9]+',
                   'THEAP']

def is_column_keyword(keyword):
    for c in COLUMN_KEYWORDS:
        if re.match(c, keyword) is not None:
            return True
    return False

def _write_fits(f, data, meta, **kwargs):
    try:
        from astropy.io import fits
    except ImportError:
        try:
            import pyfits as fits
        except ImportError:
            raise ValueError("Either pyfits or astropy is required for 'fits'"
                             " format")

    table_hdu = fits.BinTableHDU(data)

    for key, value in meta.items():
        if is_column_keyword(key.upper()) or key.upper() in REMOVE_KEYWORDS:
            warn("Meta-data keyword {0} will be ignored since it "
                 "conflicts with a FITS reserved keyword".format(key),
                 RuntimeWarning)

        if isinstance(value, list):
            for item in value:
                try:
                    table_hdu.header.append((key, item))
                except ValueError:
                    warn("Attribute `{0}` of type {1} cannot be written "
                         "to FITS files - skipping".format(key, type(value)),
                         RuntimeWarning)
        else:
            try:
                table_hdu.header[key] = value
            except ValueError:
                warn("Attribute `{0}` of type {1} cannot be written to "
                     "FITS files - skipping".format(key, type(value)),
                     RuntimeWarning)

    table_hdu.writeto(f)

# All writers =============================================================== #
WRITERS = {'csv': _write_csv,
           'salt2': _write_salt2,
           'snana': _write_snana,
           'json': _write_json,
           'fits': _write_fits}

def write_lc(data, fname, meta=None, fmt='csv', **kwargs):
    """Write light curve data.

    Parameters
    ----------
    data : `~numpy.ndarray` or dict
        Data.
    fname : str
        Filename.
    meta : dict, optional
        A (possibly empty) dictionary of metadata. Default is None.
    fmt : {'csv', 'salt2', 'snana', 'json', 'fits'}, optional
        Format of file. Default is 'csv'. 'salt2' is the new format available
        in snfit version >= 2.3.0.
    delim : str, optional
        **[csv only]** Character used to separate entries on a line.
        Default is ' '.
    metachar : str, optional
        **[csv only]** Metadata designator. Default is '@'.
    raw : bool, optional
        **[salt2, snana]** By default, the SALT2 and SNANA writers rename
        some metadata keys and column names in order to comply with what
        snfit and SNANA expect. Set to True to override this.
        Default is False.
    pedantic : bool, optional
        **[salt2, snana]** If True, check that output column names and header
        keys comply with expected formatting, and raise a ValueError if not.
        It is probably a good idea to set to False when raw is True.
        Default is True. 
    """

    if fmt not in WRITERS:
        raise ValueError("Writer not defined for format '{}'. Options: "
                         .format(fmt) + ", ".join(WRITERS.keys()))

    if not isinstance(data, np.ndarray):
        data = dict_to_array(data)

    if meta is None:
        meta = {}

    with open(fname, 'wb') as f:
        WRITERS[fmt](f, data, meta, **kwargs)
