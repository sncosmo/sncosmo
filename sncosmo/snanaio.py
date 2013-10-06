try:
    from collections import OrderedDict as odict
except ImportError:
    try:
        from astropy.utils import OrderedDict as odict
    except:
        odict = dict
import numpy as np
from astropy.io import fits

from .lcio import dict_to_array

__all__ = ['read_snana_ascii', 'read_snana_fits', 'read_snana_simlib']

def read_snana_fits(head_file, phot_file):
    """Read the SNANA FITS format: two FITS files jointly representing
    metadata and photometry for a set of SNe.

    Parameters
    ----------
    head_file : str
        Filename of "HEAD" ("header") FITS file.
    phot_file : str
        Filename of "PHOT" ("photometry") FITS file.

    Returns
    -------
    sne : list of dict
        Each item in the list is a dictionary containing a 'meta' and a 'data'
        keyword. for each item in the list, ``item['meta']`` is a dictionary
        of metadata, and ``item['data']`` is a ``~numpy.ndarray`` of the 
        photometric data.

    Examples
    --------

    >>> sne = read_snana_fits('HEAD.fits', 'PHOT.fits')
    >>> for sn in sne:
    ...     sn['meta']  # Metadata in an OrderedDict.
    ...     sn['data']  # Photometry data in a structured array.

    """

    sne = []

    # Get metadata for all the SNe
    head_data, head_hdr = fits.getdata(head_file, 1, header=True)
    phot_data, phot_hdr = fits.getdata(phot_file, 1, header=True)

    # Get views of the data as numpy arrays
    head_data = head_data.view(np.ndarray)
    phot_data = phot_data.view(np.ndarray)

    # Loop over SNe in HEAD file
    for i in range(len(head_data)):
        meta = odict(zip(head_data.dtype.names, head_data[i]))

        # convert a few keys that are the wrong type
        meta['SNID'] = int(meta['SNID'])

        j0 = head_data['PTROBS_MIN'][i] - 1 
        j1 = head_data['PTROBS_MAX'][i]
	data = phot_data[j0:j1]
	sne.append({'meta': meta, 'data': data})

    return sne


def read_snana_ascii(fname, default_tablename=None, array=True):
    """Read an SNANA-format ascii file.

    Such files may contain metadata lines and one or more tables. See Notes
    for a summary of the format.

    Parameters
    ----------
    fname : str
        Filename of object to read.
    default_tablename : str, optional
        Default tablename, or the string that indicates a table row, when
        a table starts with 'NVAR:' rather than 'NVAR_TABLENAME:'.
    array : bool, optional
        If True, each table is converted to a numpy array. If False, each
        table is a dictionary of lists (each list is a column). Default is
        True.

    Returns
    -------
    meta : `OrderedDict` or `dict`
        Metadata.
    tables : dictionary of multiple `~numpy.ndarray` or `dict`.
        Tables, indexed by table name.

    Notes
    -----
    The file can contain one or more tables, as well as optional metadata.
    Here is an example of the expected format::

        META1: a
        META2: 6
        NVAR_SN: 3
        VARNAMES: A B C
        SN: 1 2.0 x
        SN: 4 5.0 y
        SN: 5 8.2 z

    Behavior:

    * Any strings ending in a colon (:) are treated as keywords.
    * The start of a new table is indicated by a keyword starting with
      'NVAR'.
    * If the 'NVAR' is followed by an underscore (e.g., 'NVAR_TABLENAME'),
      then 'TABLENAME' is taken to be the name of the table. Otherwise the
      user *must specify* a ``default_tablename``.  This is because data
      rows are identified by the tablename.
    * After a keyword starting with 'NVAR', the next keyword must be
      'VARNAMES'. The strings following give the column names.
    * Any other keywords anywhere in the file are treated as metadata. The
      first string after the keyword is treated as the value for that keyword.
    * **Note:** Newlines are treated as equivalent to spaces; they do not
      indicate a new row.

    Examples
    --------
    If the above example is contained in the file 'data.txt'::

        >>> meta, tables = read_snana_ascii('data.txt')
        >>> meta
        OrderedDict([('META1', 'a'), ('META2', 6)])
        >>> tables['SN']
        array([(1, 2.0, 'x'), (4, 5.0, 'y'), (5, 8.2, 'z')], 
              dtype=[('A', '<i8'), ('B', '<f8'), ('C', '|S1')])
        >>> tables['SN']['A']
        array([1, 4, 5])

    If the file had an 'NVAR' keyword rather than 'NVAR_SN', for example::

        NVAR: 3
        VARNAMES: A B C
        SN: 1 2.0 x
        SN: 4 5.0 y
        SN: 5 8.2 z

    it can be read by supplying a default table name::

        >>> meta, tables = read_snana_ascii('data.txt', default_tablename='SN')
        >>> tables['SN']
        array([(1, 2.0, 'x'), (4, 5.0, 'y'), (5, 8.2, 'z')], 
              dtype=[('A', '<i8'), ('B', '<f8'), ('C', '|S1')])

    """

    meta = odict() # initialize structure to hold metadata.
    tables = {} # initialize structure to hold data.

    infile = open(fname, 'r')
    words = infile.read().split()
    infile.close()
    i = 0
    nvar = None
    tablename = None
    while i < len(words):
        word = words[i]

        # If the word starts with 'NVAR', we are starting a new table.
        if word.startswith('NVAR'):
            nvar = int(words[i + 1])

            #Infer table name. The name will be used to designate a data row.
            if '_' in word:
                pos = word.find('_') + 1
                tablename = word[pos:].rstrip(':')
            elif default_tablename is not None:
                tablename = default_tablename
            else:
                raise ValueError(
                    'table name must be given as part of NVAR keyword so '
                    'that rows belonging to this table can be identified. '
                    'Alternatively, supply the default_tablename keyword.')
            table = odict()
            tables[tablename] = table

            i += 2

        # If the word starts with 'VARNAMES', the following `nvar` words
        # define the column names of the table.
        elif word.startswith('VARNAMES'):

            # Check that nvar is defined and that no column names are defined
            # for the current table.
            if nvar is None or len(table) > 0:
                raise Exception('NVAR must directly precede VARNAMES')

            # Read the column names
            for j in range(i + 1, i + 1 + nvar):
                table[words[j]] = []
            i += nvar + 1

        # If the word matches the current tablename, we are reading a data row.
        elif word.rstrip(':') == tablename:
            for j, colname in enumerate(table.keys()):
                table[colname].append(words[i + 1 + j])
            i += nvar + 1
        
        # Otherwise, we are reading metadata or some comment
        # If the word ends with ":", it is metadata.
        elif word[-1] == ':':
            name = word[:-1]  # strip off the ':'
            if len(words) >= i + 2:
                try:
                    val = int(words[i + 1])
                except ValueError:
                    try:
                        val = float(words[i + 1])
                    except ValueError:
                        val = words[i + 1]
                meta[name] = val
            else: 
                meta[name] = None
            i += 2
        else:
            # It is some comment; continue onto next word.
            i += 1

    # All values in each column are currently strings. Convert to int or
    # float if possible.
    for table in tables.values():
        for colname, values in table.iteritems():
            try:
                table[colname] = [int(val) for val in values]
            except ValueError:
                try:
                    table[colname] = [float(val) for val in values]
                except ValueError:
                    pass

    # Convert tables to ndarrays, if requested.
    if array:
        for tablename in tables.keys():
            tables[tablename] = dict_to_array(tables[tablename])

    return meta, tables


def read_snana_ascii_multi(fnames, default_tablename=None, array=True):
    """Like ``read_snana_ascii()``, but read from multiple files containing
    the same tables and glue results together into big tables.

    Parameters
    ----------
    fnames : list of str
        List of filenames.

    Returns
    -------
    tables : dictionary of `numpy.ndarray`s or dictionaries.

    Notes
    -----
    Reading a lot of large text files in Python can become slow. Try caching
    results with cPickle, numpy.save, numpy.savez if it makes sense for your
    application.

    Examples
    --------
    >>> tables = read_snana_ascii_multi(['data1.txt', 'data1.txt'])
    >>> tables.keys()
    ['SN']
    >>> tables['SN'].dtype.names
    ('A', 'B', 'C')
    >>> tables['SN']['A']
    array([1, 4, 5, 1, 4, 5])
   
    """

    compiled_tables = {}
    for fname in fnames:

        meta, tables = read_snana_ascii(fname,
                                        default_tablename=default_tablename,
                                        array=False)
        for key, table in tables.iteritems():

            # If we already have a table with this key,
            # append this table to it.
            if key in compiled_tables:
                colnames = compiled_tables[key].keys()
                for colname in colnames:
                    compiled_tables[key][colname].extend(table[colname])

            # Otherwise, start a table
            else:
                compiled_tables[key] = table

    if array:
        for key in compiled_tables:
            compiled_tables[key] = dict_to_array(compiled_tables[key])

    return compiled_tables

def _parse_meta_from_line(line):
    """Return dictionary from key, value pairs on a line. Helper function for
    snana_read_simlib."""

    meta = odict()

    # Find position of all the colons
    colon_pos = []
    i = line.find(':')
    while i != -1:
        colon_pos.append(i)
        i = line.find(':', i+1)

    # Find position of start of words before colons
    key_pos = []
    for i in colon_pos:
        j = line.rfind(' ', 0, i)
        key_pos.append(j+1)

    # append an extra key position so that we know when to end the last value.
    key_pos.append(len(line))

    # get the keys, values based on positions above.
    for i in range(len(colon_pos)):
        key = line[key_pos[i]: colon_pos[i]]
        val = line[colon_pos[i]+1: key_pos[i+1]].strip()
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        meta[key] = val
    return meta


def read_snana_simlib(fname):
    """Read an SNANA 'simlib' (simulation library) ascii file.

    Parameters
    ----------
    fname : str
        Filename.

    Returns
    -------
    meta : `OrderedDict`
        Global meta data, not associated with any one LIBID.
    observation_sets : `OrderedDict` of `astropy.table.Table`
        keys are LIBIDs, values are observation sets.

    Notes
    -----
    * Anything following '#' on each line is ignored as a comment.
    * Keywords are space separated strings ending wth a colon.
    * If a line starts with 'LIBID:', the following lines are associated
      with the value of LIBID, until 'END_LIBID:' is encountered.
    * While reading a given LIBID, lines starting with 'S' or 'T'
      keywords are assumed to contain 12 space-separated values after
      the keyword. These are (1) MJD, (2) IDEXPT, (3) FLT, (4) CCD GAIN,
      (5) CCD NOISE, (6) SKYSIG, (7) PSF1, (8) PSF2, (9) PSF 2/1 RATIO,
      (10) ZPTAVG, (11) ZPTSIG, (12) MAG.
    * Other lines inside a 'LIBID:'/'END_LIBID:' pair are treated as metadata
      for that LIBID.
    * Any other keywords outside a 'LIBID:'/'END_LIBID:' pair are treated
      as global header keywords and are returned in the `meta` dictionary.
    
    Examples
    --------
    >>> meta, obs_sets = sncosmo.read_snana_simlib('DES_hybrid_griz.SIMLIB')
    >>> len(obs_sets)
    5
    >>> obs_sets.keys()  # LIBID for each observation set.
    [0, 1, 2, 3, 4] 
    >>> obs_sets[0].meta  # inspect observation set with LIBID=0
    OrderedDict([('LIBID', 0), ('RA', 52.5), ('DECL', -27.5), ('NOBS', 161),
                 ('MWEBV', 0.0), ('PIXSIZE', 0.27)])
    >>> obs_sets[0].colnames
    ['SEARCH', 'MJD', 'IDEXPT', 'FLT', 'CCD_GAIN', 'CCD_NOISE', 'SKYSIG',
     'PSF1', 'PSF2', 'PSFRATIO', 'ZPTAVG', 'ZPTSIG', 'MAG']

    """

    from astropy.table import Table

    COLNAMES = ['SEARCH', 'MJD', 'IDEXPT', 'FLT', 'CCD_GAIN', 'CCD_NOISE',
                'SKYSIG', 'PSF1', 'PSF2', 'PSFRATIO', 'ZPTAVG', 'ZPTSIG', 'MAG']
    SPECIAL = ['FIELD', 'TELESCOPE', 'PIXSIZE'] # not used yet... if present in 
                                                # header, add to table.

    meta = odict()  # global metadata
    observation_sets = odict() # dictionary of tables indexed by LIBID

    reading_obsset = False
    with open(fname, 'r') as infile:
        for line in infile.readlines():

            # strip comments
            idx = line.find('#')
            if idx != -1:
                line = line[0:idx]

            # split on spaces.
            words = line.split()
            if len(words) == 0: continue

            # are we currently reading an obsset?
            if not reading_obsset:
                if line[0:6] == 'LIBID:':
                    reading_obsset = True
                    current_meta = _parse_meta_from_line(line)
                    current_data = odict([(key, []) for key in COLNAMES])
                else:
                    meta.update(_parse_meta_from_line(line))

            else:
                if line[0:10] == 'END_LIBID:':
                    reading_obsset = False
                    observation_sets[current_meta['LIBID']] = \
                        Table(current_data, meta=current_meta)
                elif line[0:2] in ['S:', 'T:']:
                    words = line.split()
                    for colname, val in [
                        ('SEARCH', words[0] == 'S:'),
                        ('MJD', float(words[1])),
                        ('IDEXPT', int(words[2])),
                        ('FLT', words[3]),
                        ('CCD_GAIN', float(words[4])),
                        ('CCD_NOISE', float(words[5])),
                        ('SKYSIG', float(words[6])),
                        ('PSF1', float(words[7])),
                        ('PSF2', float(words[8])),
                        ('PSFRATIO', float(words[9])),
                        ('ZPTAVG', float(words[10])),
                        ('ZPTSIG', float(words[11])),
                        ('MAG', float(words[12]))]:
                        current_data[colname].append(val)
                else:
                    current_meta.update(_parse_meta_from_line(line))

    return meta, observation_sets
