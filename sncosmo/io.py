# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions for supernova light curve I/O"""

from __future__ import print_function

from warnings import warn
import math
import os
import sys
import re
import json
from collections import OrderedDict

import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy import wcs
from astropy.extern import six

from .utils import dict_to_array
from .bandpasses import get_bandpass

__all__ = ['read_lc', 'write_lc', 'load_example_data', 'read_griddata_ascii',
           'read_griddata_fits', 'write_griddata_ascii', 'write_griddata_fits']


def _stripcomment(line, char='#'):
    pos = line.find(char)
    if pos == -1:
        return line
    else:
        return line[:pos]


def _cast_str(s):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s.strip()


def read_griddata_ascii(name_or_obj):
    """Read 2-d grid data from a text file.

    Each line has values `x0 x1 y`. Space separated.
    x1 values are only read for first x0 value. Others are assumed
    to match.

    Parameters
    ----------
    name_or_obj : str or file-like object

    Returns
    -------
    x0 : numpy.ndarray
        1-d array.
    x1 : numpy.ndarray
        1-d array.
    y : numpy.ndarray
        2-d array of shape (len(x0), len(x1)).
    """

    if isinstance(name_or_obj, six.string_types):
        f = open(name_or_obj, 'r')
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
        if len(stripped_line) == 0:
            continue
        x0_tmp, x1_tmp, y_tmp = map(float, stripped_line.split())
        if x0_current is None:
            x0_current = x0_tmp  # Initialize first time

        # If there is a new x0 value, ingest the old one and reset values
        if x0_tmp != x0_current:
            x0.append(x0_current)
            if x1 is None:
                x1 = x1_current
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


def read_griddata_fits(name_or_obj, ext=0):
    """Read a multi-dimensional grid of data from a FITS file, where the
    grid coordinates are encoded in the FITS-WCS header keywords.

    Parameters
    ----------
    name_or_obj : str or file-like object

    Returns
    -------
    x0, x1, ... : `~numpy.ndarray`
        1-d arrays giving coordinates of grid. The number of these arrays will
        depend on the dimension of the data array. For example, if the data
        have two dimensions, a total of three arrays will be returned:
        ``x0, x1, y``, with ``x0`` giving the coordinates of the first axis
        of ``y``. If the data have three dimensions, a total of four arrays
        will be returned: ``x0, x1, x2, y``, and so on with higher dimensions.
    y : `~numpy.ndarray`
        n-d array of shape ``(len(x0), len(x1), ...)``. For three dimensions
        for example, the value at ``y[i, j, k]`` corresponds to coordinates
        ``(x0[i], x1[j], x2[k])``.
    """

    hdulist = fits.open(name_or_obj)
    w = wcs.WCS(hdulist[ext].header)
    y = hdulist[ext].data

    # get abcissa values (coordinates at grid values)
    xs = []
    for i in range(y.ndim):
        j = y.ndim - i  # The i-th axis (in Python) corresponds to FITS AXISj
        coords = np.zeros((y.shape[i], y.ndim), dtype=np.float32)
        coords[:, j-1] = np.arange(y.shape[i])
        x = w.wcs_pix2world(coords, 0)[:, j-1]
        xs.append(x)

    hdulist.close()

    return tuple(xs) + (y,)


def write_griddata_ascii(x0, x1, y, name_or_obj):
    """Write 2-d grid data to a text file.

    Each line has values `x0 x1 y`. Space separated.

    Parameters
    ----------
    x0 : numpy.ndarray
        1-d array.
    x1 : numpy.ndarray
        1-d array.
    y : numpy.ndarray
        2-d array of shape (len(x0), len(x1)).
    name_or_obj : str or file-like object
        Filename to write to or open file.
    """

    if isinstance(name_or_obj, six.string_types):
        f = open(name_or_obj, 'w')
    else:
        f = name_or_obj

    for j in range(len(x0)):
        for i in range(len(x1)):
            f.write("{0:.7g} {1:.7g} {2:.7g}\n".format(x0[j], x1[i], y[j, i]))

    if isinstance(name_or_obj, six.string_types):
        f.close()


def write_griddata_fits(x0, x1, y, name_or_obj):
    """Write a 2-d grid of data to a FITS file

    The grid coordinates are encoded in the FITS-WCS header keywords.

    Parameters
    ----------
    x0 : numpy.ndarray
        1-d array.
    x1 : numpy.ndarray
        1-d array.
    y : numpy.ndarray
        2-d array of shape (len(x0), len(x1)).
    name_or_obj : str or file-like object
        Filename to write to or open file.
    """

    d0, d1 = np.ediff1d(x0), np.ediff1d(x1)
    if not (np.allclose(d0, d0[0]) and np.allclose(d1, d1[0])):
        raise ValueError('grid must be regularly spaced in both x0 and x1')
    if not (len(x0), len(x1)) == y.shape:
        raise ValueError('length of x0 and x1 do not match shape of y')

    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [1, 1]
    w.wcs.crval = [x1[0], x0[0]]
    w.wcs.cdelt = [d1[0], d0[0]]
    hdu = fits.PrimaryHDU(y, header=w.to_header())
    hdu.writeto(name_or_obj)


# -----------------------------------------------------------------------------
# Reader: ascii
def _read_ascii(f, delim=None, metachar='@', commentchar='#'):

    meta = OrderedDict()
    colnames = []
    cols = []
    readingdata = False
    for line in f:

        # strip leading & trailing whitespace, newline, and comments
        line = line.strip()
        pos = line.find(commentchar)
        if pos > -1:
            line = line[:pos]
        if len(line) == 0:
            continue

        if not readingdata:
            # Read metadata
            if line[0] == metachar:
                pos = line.find(' ')  # Find first space.
                if pos in [-1, 1]:  # Space must exist and key must exist.
                    raise ValueError('Incorrectly formatted metadata line: ' +
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

    data = OrderedDict(zip(colnames, cols))
    return meta, data


# -----------------------------------------------------------------------------
# Reader: salt2

def _expand_bands(band_list, meta):
    """Given a list containing band names, return a list of Bandpass objects"""

    # Treat dependent bandpasses based on metadata contents
    # TODO: need a way to figure out which bands are position dependent!
    #       for now, we assume *all* or none are.
    if "X_FOCAL_PLANE" in meta and "Y_FOCAL_PLANE" in meta:
        r = math.sqrt(meta["X_FOCAL_PLANE"]**2 + meta["Y_FOCAL_PLANE"]**2)

        # map name to object for unique bands
        name_to_band = {name: get_bandpass(name, r)
                        for name in set(band_list)}

        return [name_to_band[name] for name in band_list]

    else:
        # For other bandpasses, get_bandpass will return the same object
        # on each call, so just use it directly.
        return [get_bandpass(name) for name in band_list]


def _read_salt2(name_or_obj, read_covmat=False, expand_bands=False):
    """Read a new-style SALT2 file.

    Such a file has metadata on lines starting with '@' and column names
    on lines starting with '#' and containing a ':' after the column name.
    There is optionally a line containing '#end' before the start of data.
    """

    if isinstance(name_or_obj, six.string_types):
        f = open(name_or_obj, 'r')
    else:
        f = name_or_obj

    meta = OrderedDict()
    colnames = []
    cols = []
    readingdata = False
    for line in f:

        # strip leading & trailing whitespace & newline
        line = line.strip()
        if len(line) == 0:
            continue

        if not readingdata:
            # Read metadata
            if line[0] == '@':
                pos = line.find(' ')  # Find first space.
                if pos in [-1, 1]:  # Space must exist and key must exist.
                    raise ValueError('Incorrectly formatted metadata line: ' +
                                     line)
                meta[line[1:pos]] = _cast_str(line[pos:])
                continue

            # Read header line
            if line[0] == '#':
                pos = line.find(':')
                if pos in [-1, 1]:
                    continue  # comment line
                colname = line[1:pos].strip()
                if colname == 'end':
                    continue
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
        if len(line) == 0:
            continue

        # Now we're reading data
        items = line.split()
        for col, item in zip(cols, items):
            col.append(_cast_str(item))

    if isinstance(name_or_obj, six.string_types):
        f.close()

    # read covariance matrix file, if requested and present
    if read_covmat and 'COVMAT' in meta:
        fname = os.path.join(os.path.dirname(f.name), meta['COVMAT'])

        # use skiprows=1 because first row has array dimensions
        fluxcov = np.loadtxt(fname, skiprows=1)

        # asethetics: capitalize 'Fluxcov' to match salt2 colnames
        # such as 'Fluxerr'
        colnames.append('Fluxcov')
        cols.append(fluxcov)

    data = OrderedDict(zip(colnames, cols))

    if expand_bands:
        data['Filter'] = _expand_bands(data['Filter'], meta)

    return meta, data


# -----------------------------------------------------------------------------
# Reader: salt2-old

def _read_salt2_old(dirname, filenames=None):
    """Read old-style SALT2 files from a directory.

    A file named 'lightfile' must exist in the directory.
    """

    # Get list of files in directory.
    if not (os.path.exists(dirname) and os.path.isdir(dirname)):
        raise IOError("Not a directory: '{0}'".format(dirname))
    dirfilenames = os.listdir(dirname)

    # Read metadata from lightfile.
    if 'lightfile' not in dirfilenames:
        raise IOError("no lightfile in directory: '{0}'".format(dirname))
    with open(os.path.join(dirname, 'lightfile'), 'r') as lightfile:
        meta = OrderedDict()
        for line in lightfile.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            try:
                key, val = line.split()
            except ValueError:
                raise ValueError('expected space-separated key value pairs in '
                                 'lightfile: {0}'
                                 .format(os.path.join(dirname, 'lightfile')))
            meta[key] = _cast_str(val)

    # Get list of filenames to read.
    if filenames is None:
        filenames = dirfilenames
    if 'lightfile' in filenames:
        filenames.remove('lightfile')  # We already read the lightfile.
    fullfilenames = [os.path.join(dirname, f) for f in filenames]

    # Read data from files.
    data = None
    for fname in fullfilenames:
        with open(fname, 'r') as f:
            filemeta, filedata = _read_salt2(f)

        # Check that all necessary file metadata was defined.
        if not ('INSTRUMENT' in filemeta and 'BAND' in filemeta and
                'MAGSYS' in filemeta):
            raise ValueError('not all necessary global keys (INSTRUMENT, '
                             'BAND, MAGSYS) are defined in file {0}'
                             .format(fname))

        # Add the instrument/band to the file data, in anticipation of
        # aggregating it with other files.

        # PY3: next(iter(filedata.vlues()))
        firstcol = six.next(six.itervalues(filedata))
        data_length = len(firstcol)
        filter_name = '{0}::{1}'.format(filemeta.pop('INSTRUMENT'),
                                        filemeta.pop('BAND'))
        filedata['Filter'] = data_length * [filter_name]
        filedata['MagSys'] = data_length * [filemeta.pop('MAGSYS')]

        # If this if the first file, initialize data lists, otherwise if keys
        # match, append this file's data to the main data.
        if data is None:
            data = filedata
        elif set(filedata.keys()) == set(data.keys()):
            for key in data:
                data[key].extend(filedata[key])
        else:
            raise ValueError('column names do not match between files')

        # Append any extra metadata in this file to the master metadata.
        if len(filemeta) > 0:
            meta[filter_name] = filemeta

    return meta, data


# -----------------------------------------------------------------------------
# Reader: json
def _read_json(f):
    t = json.load(f)

    # Encode data keys as ascii rather than UTF-8 so that they can be
    # used as numpy structured array names later.
    d = {}
    for key, value in t['data'].items():
        d[key] = value
    return t['meta'], d


# -----------------------------------------------------------------------------
# All readers
READERS = {'ascii': _read_ascii,
           'json': _read_json,
           'salt2': _read_salt2,
           'salt2-old': _read_salt2_old}


def read_lc(file_or_dir, format='ascii', **kwargs):
    """Read light curve data for a single supernova.

    Parameters
    ----------
    file_or_dir : str
        Filename (formats 'ascii', 'json', 'salt2') or directory name
        (format 'salt2-old'). For 'salt2-old' format, directory must contain
        a file named 'lightfile'. All other files in the directory are
        assumed to be photometry files, unless the `filenames` keyword argument
        is set.
    format : {'ascii', 'json', 'salt2', 'salt2-old'}, optional
        Format of file. Default is 'ascii'. 'salt2' is the new format available
        in snfit version >= 2.3.0.
    read_covmat : bool, optional
        **[salt2 only]** If True, and if a ``COVMAT`` keyword is present in
        header, read the covariance matrix from the filename specified
        by ``COVMAT`` (assumed to be in the same directory as the lightcurve
        file) and include it as a column named ``Fluxcov`` in the returned
        table. Default is False.

        *New in version 1.5.0*

    expand_bands : bool, optional
        **[salt2 only]** If True, convert band names into equivalent Bandpass
        objects. This is particularly useful for position-dependent
        bandpasses in the salt2 file format: the position information is
        read from the header and used when creating the bandpass objects.

        *New in version 1.5.0*

    delim : str, optional
        **[ascii only]** Used to split entries on a line. Default is `None`.
        Extra whitespace is ignored.
    metachar : str, optional
        **[ascii only]** Lines whose first non-whitespace character is
        `metachar` are treated as metadata lines, where the key and value
        are split on the first whitespace. Default is ``'@'``
    commentchar : str, optional
        **[ascii only]** One-character string indicating a comment. Default is
        '#'.
    filenames : list, optional
        **[salt2-old only]** Only try to read the given filenames as
        photometry files. Default is to try to read all files in directory.

    Returns
    -------
    t : astropy `~astropy.table.Table`
        Table of data. Metadata (as an `OrderedDict`) can be accessed via
        the ``t.meta`` attribute. For example: ``t.meta['key']``. The key
        is case-sensitive.

    Examples
    --------

    Read an ascii format file that includes metadata (``StringIO``
    behaves like a file object):

    >>> from astropy.extern.six import StringIO
    >>> f = StringIO('''
    ... @id 1
    ... @RA 36.0
    ... @description good
    ... time band flux fluxerr zp zpsys
    ... 50000. g 1. 0.1 25. ab
    ... 50000.1 r 2. 0.1 25. ab
    ... ''')
    >>> t = read_lc(f, format='ascii')
    >>> print(t)
      time  band flux fluxerr  zp  zpsys
    ------- ---- ---- ------- ---- -----
    50000.0    g  1.0     0.1 25.0    ab
    50000.1    r  2.0     0.1 25.0    ab
    >>> t.meta
    OrderedDict([('id', 1), ('RA', 36.0), ('description', 'good')])

    """

    try:
        readfunc = READERS[format]
    except KeyError:
        raise ValueError("Reader not defined for format {0!r}. Options: "
                         .format(format) + ", ".join(READERS.keys()))

    if format == 'salt2-old':
        meta, data = readfunc(file_or_dir, **kwargs)
    elif isinstance(file_or_dir, six.string_types):
        with open(file_or_dir, 'r') as f:
            meta, data = readfunc(f, **kwargs)
    else:
        meta, data = readfunc(file_or_dir, **kwargs)

    return Table(data, meta=meta)


# =========================================================================== #
# Writers                                                                     #
# =========================================================================== #

# -----------------------------------------------------------------------------
# Writer: ascii
def _write_ascii(f, data, meta, **kwargs):

    delim = kwargs.get('delim', ' ')
    metachar = kwargs.get('metachar', '@')

    if meta is not None:
        for key, val in six.iteritems(meta):
            f.write('{0}{1}{2}{3}\n'.format(metachar, key, delim, str(val)))

    keys = data.dtype.names
    length = len(data)

    f.write(delim.join(keys))
    f.write('\n')
    for i in range(length):
        f.write(delim.join([str(data[key][i]) for key in keys]))
        f.write('\n')

# -----------------------------------------------------------------------------
# Writer: salt2
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
        for key, val in six.iteritems(meta):
            if not raw:
                key = key.upper()
                key = KEY_TO_SALT2KEY_META.get(key, key)
            f.write('@{0} {1}\n'.format(key, str(val)))

    keys = data.dtype.names
    length = len(data)

    # Write column names
    keys_as_written = []
    for key in keys:
        if not raw:
            key = key.capitalize()
            key = KEY_TO_SALT2KEY_COLUMN.get(key, key)
        f.write('#{0} :\n'.format(key))
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


# -----------------------------------------------------------------------------
# Writer: snana

KEY_TO_SNANAKEY_COLUMN = {
    'TIME': 'MJD',
    'DATE': 'MJD',
    'FILTER': 'FLT',
    'BAND': 'FLT',
    'FLUX': 'FLUXCAL',
    'FLUXERR': 'FLUXCALERR',
    'ZP': 'ZPT',
    'ZEROPOINT': 'ZPT'}
KEY_TO_SNANAKEY_META = {
    'DEC': 'DECL'}
SNANA_REQUIRED_META = ['RA', 'DECL', 'SURVEY', 'FILTERS', 'MWEBV']
SNANA_REQUIRED_COLUMN = ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR', 'ZPT']


def _write_snana(f, data, meta, **kwargs):

    raw = kwargs.get('raw', False)
    pedantic = kwargs.get('pedantic', True)

    # Write metadata
    keys_as_written = []
    if meta is not None:
        for key, val in six.iteritems(meta):
            if not raw:
                key = key.upper()
                key = KEY_TO_SNANAKEY_META.get(key, key)
            f.write('{0}: {1}\n'.format(key, str(val)))
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
            'NOBS: {0:d}\n'
            'NVAR: {1:d}\n'
            'VARLIST: {2}\n'
            .format(length, len(keys), ' '.join(keys_to_write)))

    # Write data
    for i in range(length):
        f.write('OBS: ')
        f.write(' '.join([str(data[key][i]) for key in keys]))
        f.write('\n')


# -----------------------------------------------------------------------------
# Writer: json
def _write_json(f, data, meta, **kwargs):

    # Build a dictionary of pure-python objects
    output = OrderedDict([('meta', meta),
                          ('data', OrderedDict())])
    for key in data.dtype.names:
        output['data'][key] = data[key].tolist()
    json.dump(output, f)
    del output


# -----------------------------------------------------------------------------
# All writers
WRITERS = {'ascii': _write_ascii,
           'salt2': _write_salt2,
           'snana': _write_snana,
           'json': _write_json}


def write_lc(data, fname, format='ascii', **kwargs):
    """Write light curve data.

    Parameters
    ----------
    data : `~astropy.table.Table`
        Light curve data.
    fname : str
        Filename.
    format : {'ascii', 'salt2', 'snana', 'json'}, optional
        Format of file. Default is 'ascii'. 'salt2' is the new format available
        in snfit version >= 2.3.0.
    delim : str, optional
        **[ascii only]** Character used to separate entries on a line.
        Default is ' '.
    metachar : str, optional
        **[ascii only]** Metadata designator. Default is '@'.
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

    if format not in WRITERS:
        raise ValueError("Writer not defined for format {0!r}. Options: "
                         .format(format) + ", ".join(WRITERS.keys()))
    if isinstance(data, Table):
        meta = data.meta
        data = np.asarray(data)
    else:
        meta = OrderedDict()
        if not isinstance(data, np.ndarray):
            data = dict_to_array(data)
    with open(fname, 'w') as f:
        WRITERS[format](f, data, meta, **kwargs)


def load_example_data():
    """
    Load an example photometric data table.

    Returns
    -------
    data : `~astropy.table.Table`
    """
    from astropy.utils.data import get_pkg_data_filename
    filename = get_pkg_data_filename(
        'data/examples/example_photometric_data.dat')
    return read_lc(filename, format='ascii')
