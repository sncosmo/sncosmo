"""Functions for reading and writing measured light curve data to 
SALT2-format files"""

import os
import math
from collections import OrderedDict
import numpy as np

__all__ = ["read", "write", "readdir", "writedir"]

# Reading Metadata
# ----------------
# Names are converted to lowercase, then the following lookup table is used.
SALTMETA_TO_META = {
    'redshift': 'z',
    'z_heliocentric': 'z_helio'}
# Metadata whose names (after above operations) match these are converted to
# float.  Any others are left as strings.
FLOAT_META = ['z', 'z_helio', 'z_cmb', 'mwebv']

# Writing Metadata
# ----------------
# Names are converted to uppercase, then the following lookup table is used
# for both correct capitalization and translation.
META_TO_SALTMETA_OLD = {
    'Z': 'Redshift',      # 'Redshift' can actually be any case.
    'REDSHIFT': 'Redshift',
    'Z_HELIO': 'z_heliocentric',
    'Z_HELIOCENTRIC': 'z_heliocentric',
    'Z_CMB': 'z_cmb'}
META_TO_SALTMETA_NEW = {
    'Z': 'REDSHIFT',              # Not sure if this is used.
    'Z_HELIOCENTRIC': 'Z_HELIO',
    'MAGSYS': 'MagSys',
    'Z_SOURCE': 'z_source'} 


# Reading Columns
# ---------------
# Names are converted to lowercase, then the following lookup table is used.
SALTCOLS_TO_COLS = {
    'filter': 'band'}
# Columns whose names match these (after conversion to lowercase and lookup
# table) are left as strings. All others are converted to float.
STRING_COLS = ['band', 'magsys', 'instrument', 'filter']

# Writing Columns
# ---------------
# Names are capitalized (lowercase except first
# letter capitalized) and then the following lookup table is used.
COLS_TO_SALTCOLS = {
    'Fluxpsf': 'FluxPsf',
    'Fluxpsferr': 'FluxPsferr',
    'Airmass': 'AirMass',
    'Zp': 'ZP',
    'Magsys': 'MagSys',
    'Band': 'Filter'}


def _rawmeta_to_meta(meta):
    """Alter an OrderedDict according to some rules."""

    newmeta = OrderedDict()
    for key, val in meta.iteritems():
        key = key.lower()
        if key in SALTMETA_TO_META:
            key = SALTMETA_TO_META[key]
        if key in FLOAT_META:
            val = float(val)
        newmeta[key] = val
    return newmeta

def _rawcols_to_cols(cols):
    """Alter OrderedDict of lists according to some rules."""

    newcols = OrderedDict()
    for key in cols:
        newkey = key.lower()
        if newkey in SALTCOLS_TO_COLS:
            newkey = SALTCOLS_TO_COLS[newkey]
        if newkey in STRING_COLS:
            newcols[newkey] = cols[key]
        else:
            newcols[newkey] = [float(val) for val in cols[key]]
    return newcols


def _dict_to_ndarray(d):
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

def _read_lightfile(filename):
    """Read lightfile (deprecated format) and return dictionary of keywords.

    Comment lines are not allowed.
    """

    lightfile = open(filename, 'r')
    metadata = OrderedDict()

    for line in lightfile.readlines():
        trimmedline = line.strip()
        if len(trimmedline) == 0: continue
        lineitems = trimmedline.split()
        if len(lineitems) > 2:
            raise ValueError('expect key value pairs in lightfile: '
                             '{}'.format(filename))
        key, val = lineitems
        metadata[key] = val

    lightfile.close()

    return metadata

def _read_dictfile(filename):
    """Read a text data file with SALT-format metadata and header tags.
    
    Such a file has metadata on lines starting with '@' and column names
    on lines starting with '#' and containing a ':' after the column name.

    Returns
    -------
    metadata : OrderedDict
        File metadata
    cols : OrderedDict of lists
        data columns
    """

    # Initialize output containers.
    metadata = OrderedDict()
    cols = OrderedDict()

    foundend = False
    f = open(filename, 'r')
    for line in f.readlines():
        line = line.strip()

        # If a blank line, just skip it.
        if len(line) == 0: continue
                  
        # If a metadata line:
        if line[0] == '@':
            pos = line.find(' ')  # Find first space in the line.

            # It must have a space, and it can't be right after the @. 
            if pos in [-1, 1]:
                raise ValueError('Incorrectly formatted metadata line: '
                                 '{}'.format(line))

            key = line[1:pos]
            val = line[pos:].strip()
            metadata[key] = val

        # Comment line, end flag, or column name.
        elif line[0] == '#':

            # Test for the end flag.
            if line[1:].strip(': ') == 'end':
                foundend = True

            # Test for comment line.
            else:
                pos = line.find(':')
                if pos in [-1, 1]: continue  # comment line
                colname = line[1:pos].strip()
                if colname == '' or ' ' in colname: continue  # comment line
                if foundend:
                    raise ValueError('Bizarre file format: header line (#) '
                                     'occuring after #end')
                cols[colname] = []
        
        # Otherwise, we are probably reading a dataline.
        elif not foundend:
            raise ValueError('data line occuring before #end flag.')
        else:
            lineitems = line.split()
            if len(lineitems) != len(cols):
                raise ValueError('line length does not match number of '
                                 'named columns in file header')
            for i, key in enumerate(cols.keys()):
                cols[key].append(lineitems[i])
    f.close()

    return metadata, cols


def read(filename):
    """Read a new-style SALT2 photometry file.

    The new style is available in snfit version >= 2.3.0.

    Parameters
    ----------
    filename : str
        
    Returns
    -------
    photdata : `numpy.ndarray`
        Supernova photometric data in a structured array.
    metadata : OrderedDict
        Supernova metadata.
    """

    metadata, cols = _read_dictfile(filename)

    # Convert metadata names and values, and column names and values
    metadata = _rawmeta_to_meta(metadata)
    cols = _rawcols_to_cols(cols)

    # Make sure there are columns called `band` and `magsys`.
    if not ('band' in cols and 'magsys' in cols):
        # Check if this is an old-style file
        if ('instrument' in metadata or 'band' in metadata or
            'magsys' in metadata):
            raise ValueError('Is this an old-style SALT file? '
                             'To read old-style files use readdir()')
        raise ValueError('some mandatory columns missing in file: ' +
                         filename)


    # NOTE: We no longer need to do any of this below

    # The filter values come to us formatted as INSTRUMENT::BAND. Parse them.
    #cols['instrument'] = []
    #cols['band'] = []
    #for filtername in cols['filter']:
    #    items = filtername.split('::')
    #    if len(items) != 2 or len(items[0]) == 0 or len(items[1]) == 0:
    #        raise ValueError('improperly formatted filter in file: ' +
    #                         filename)
    #    cols['instrument'].append(items[0])
    #    cols['band'].append(items[0])

    # We can now remove the filter column because the information is in 
    # the instrument and band columns.
    #del cols['filter']

    photdata = _dict_to_ndarray(cols)
    return photdata, metadata


def write(photdata, metadata, filename):
    """Write a new-style SALT2 photometry file.

    The new style is available in snfit version >= 2.3.0.

    Parameters
    ----------
    photdata : `numpy.ndarray` or dictionary of lists
        Photometric data, in either stuctured array, or lists representing
        columns.
    metadata : dict
        Supernova metadata.
    filename : str
        Name of file to write to.
    """
    
    outfile = open(filename, 'w')

    # Write metadata.
    for key, val in metadata.iteritems():
        key = key.upper()
        if key in META_TO_SALTMETA_NEW:
            key = META_TO_SALTMETA_NEW[key]
        outfile.write('@{} {}\n'.format(key, val))

    # Photometry data: infer the input data type and convert to ndarray
    if isinstance(photdata, dict):
        photdata = _dict_to_ndarray(photdata)
    if not isinstance(photdata, np.ndarray):
        raise ValueError('Invalid data type {0} for photometry data'
                         .format(type(photdata)))

    # Photometry data: convert colnames
    colnames = []
    for name in photdata.dtype.names:
        name = name.capitalize()
        if name in COLS_TO_SALTCOLS:
            name = COLS_TO_SALTCOLS[name]
        colnames.append(name)

    # check that the photometry data contain the fields 'filter', and 'magsys'.
    if not ('Filter' in colnames and 'MagSys' in colnames):
        raise ValueError('photometry data missing required field(s)')

    # Combine 'Instrument' and 'Band' fields into one field: 'Filter'
    #headernames[headernames.index('Instrument')] = 'Filter'
    #headernames.remove('Band')

    # Write the headers
    for colname in colnames:
        outfile.write('#{} :\n'.format(colname))
    outfile.write('#end :\n')

    # Write the data itself
    for i in range(len(photdata)):
        for name in photdata.dtype.names:
            #if name == 'band': continue
            #elif name == 'instrument':
            #    outfile.write('{}::{} '.format(photdata[i]['instrument'],
            #                                  photdata[i]['band']))
            #else:
            outfile.write('{} '.format(photdata[i][name]))
        outfile.write('\n')

    outfile.close()


def readdir(dirname, filenames=None):
    """Read old-style SALT2 photometry files for a single supernova.
    
    A file named `lightfile` must be in the directory.

    Parameters
    ----------
    dirname : str
        The directory containing the files
    filenames : list of str (optional)
        Only try to read the given filenames as photometry files. If `None`
        (default), will try to read all files in directory.

    Returns
    -------
    photdata : `numpy.ndarray`
        Structured array containing photometric data.
    metadata : OrderedDict
        Supernova metadata.
    """

    if dirname[-1] == '/': dirname = dirname[0:-1]
    if not (os.path.exists(dirname) and os.path.isdir(dirname)):
        raise IOError("Not a directory: '{}'".format(dirname))
    dirfilenames = os.listdir(dirname)

    # Get metadata from lightfile.
    if 'lightfile' not in dirfilenames:
        raise IOError("no 'lightfile' in directory: '{}'".format(name))
    metadata = _read_lightfile(dirname + '/lightfile')
    metadata = _rawmeta_to_meta(metadata)

    # Get list of filenames to read.
    if filenames is None:
        filenames = dirfilenames
    if 'lightfile' in filenames:
        filenames.remove('lightfile')  # We already read the lightfile.
    fullfilenames = [dirname + '/' + f for f in filenames]
    
    # Read photdata from files.
    allcols = None
    for filename in fullfilenames:

        filemeta, cols = _read_dictfile(filename)

        # Check that all necessary metadata was described.
        if not ('INSTRUMENT' in filemeta and
                'BAND' in filemeta and
                'MAGSYS' in filemeta):
            raise ValueError('not all necessary global keys are defined')

        # Add the metadata to columns, in anticipation of aggregating it
        # with other files.
        firstkey = cols.keys()[0]
        clen = len(cols[firstkey])  # Length of all cols.
        cols['Filter'] = \
            ['{}::{}'.format(filemeta['INSTRUMENT'], filemeta['BAND'])] * clen
        cols['MagSys'] = [filemeta['MAGSYS']] * col_len

        # Convert column names and values
        cols = _rawcols_to_cols(cols)

        # If this if the first file, initialize data lists.
        if allcols is None:
            allcols = cols
        
        # Otherwise, if keys match, append lists...
        elif set(allcols.keys()) == set(cols.keys()):
            for key in allcols: allcols[key].extend(cols[key])
        
        # and if they do not match, raise Error.
        else:
            raise ValueError('column names do not match between files')

    # Now we have all our data in lists in `allcols`. Convert this to
    # a structured numpy array.
    photdata = _dict_to_ndarray(allcols)

    return photdata, metadata


def writedir(photdata, metadata, dirname):
    """Save photometry data and metadata to old-style SALT files
    in a directory.

    Parameters
    ----------
    photdata : `numpy.ndarray` or dict
        structured array or dictionary of equal-length lists
        containing (at least) fields named 'instrument', 'band', 'magsys'
        (or some capitalization thereof).
    metadata : dict
        Dictionary containing metadata to be written to lightfile.
    dirname : string
        Path to directory.
    """

    # Photometry data: infer the input data type and convert to ndarray
    if isinstance(photdata, dict):
        photdata = _dict_to_ndarray(photdata)
    if not isinstance(photdata, np.ndarray):
        raise ValueError('Invalid data type {0} for photometry data'
                         .format(type(photdata)))

    # Make the target directory if it doesn't exist.
    if not os.path.exists(dirname): os.makedirs(dirname)
       
    # Write metadata to the "lightfile".
    outfile = open(dirname + '/lightfile', 'w')
    for key, val in metadata.iteritems():
        key = key.upper()
        if key in META_TO_SALTMETA_OLD:
            key = META_TO_SALTMETA_OLD[key]
        outfile.write('{} {}\n'.format(key, val))
    outfile.close()

    # Photometry data:
    # require that the photometry data contain the fields 'band' and 'magsys'.
    if not ('band' in photdata.dtype.names and 
            'magsys' in photdata.dtype.names):
        raise ValueError('photometry data missing required field(s)')

    # On output, each SALT photometry file has a single (band, magsys)
    # combination. Find the unique combinations of these in the table.
    index_data = photdata[['band', 'magsys']]
    unique_combos = np.unique(index_data)
    
    # Get a list of fields besides (band, magsys). These will have data columns
    fieldnames = copy.deepcopy(photdata.dtype.names)
    fieldnames.remove('band')
    fieldnames.remove('magsys')

    # Convert remaining colnames.
    colnames = []
    for name in fieldnames:
        name = name.capitalize()
        if name in COLS_TO_SALTCOLS:
            name = COLS_TO_SALTCOLS[name]
        colnames.append(name)

    # Create a photometry file for each unique combo
    for band, magsys in unique_combos:

        # `band` should be formatted like INSTRUMENT::BAND. Parse it.
        try:
            instrument, oldband = band.split('::')
        except ValueError:
            raise ValueError('band must be formatted with a double colon (::)')
        
        # Open the file.
        filename = '{}/{}_{}_{}.dat'.format(dirname, instrument, oldband,
                                            magsys)
        photfile = open(filename, 'w')
        
        # Write header of file.
        photfile.write('@INSTRUMENT {}\n'.format(instrument))
        photfile.write('@BAND {}\n'.format(oldband))
        photfile.write('@MAGSYS {}\n'.format(magsys))
        for colname in colnames:
            photfile.write('#{} :\n'.format(colname))
        photfile.write('#end :\n')

        # Find indicies of table matching this combo
        idx = ((photdata['band'] == band) &
               (photdata['magsys'] == magsys))

        matchedrows = photdata[idx]  # Get just the rows we want for this file
        for i in range(len(matchedrows)):
            for key in fieldnames:
                photfile.write('{} '.format(matchedrows[i][key]))
            photfile.write('\n')

        photfile.close()  # close this (band, magsys) file.
