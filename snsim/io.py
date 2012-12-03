from collections import OrderedDict
from astropy.table import Table

__all__ = ['read_simlib']

def read_simlib(filename):
    """Read an SNANA 'simlib' file.

    Parameters
    ----------
    filename : str

    Returns
    -------
    header : dict
        Keywords not used elsewhere.
    fields : OrderedDict
        Characteristics (ra, dec, mwebv) of each field. Entries are indexed
        by LIBID (integer). Each entry is a dictionary.
    observations : `astropy.table.Table`
        Characteristics of every observation.

    Notes
    -----
    Anything following '#' on each line is ignored as a comment.

    Lines are split into space-separated character strings.
    Character strings that end with ':' and are all uppercase are treated
    as keywords.

    Lines starting with 'S' or 'T' keywords are assumed to contain 12
    space-separated values after the keyword. These are
    (1) MJD, (2) IDEXPT, (3) FLT, (4) CCD GAIN, (5) CCD NOISE, (6)
    SKYSIG, (7) PSF1, (8) PSF2, (9) PSF 2/1 RATIO, (10) ZPTAVG, (11)
    ZPTSIG, (12) MAG.

    Any other non-blank lines are searched for keywords. Some of these
    keywords are given special treatment. These include
    'LIBID', 'TELESCOPE', 'PIXSIZE', 'RA', 'DEC', 'MWEBV'.
    
    The value of 'LIBID' is used as a unique field identifier.  The
    most recently encountered field identifier is applied to each
    observation. The values 'RA', 'DEC', and 'MWEBV' are used to set
    the field information about the most recently encountered field
    identifier.

    The most recently encountered 'TELESCOPE' and 'PIXSIZE' values are
    applied to each observation.

    Any other keywords are treated as global header keywords and are
    returned in the 'header' dictionary.
    
    Note that the 'FIELD' keyword is currently ignored. 'FIELD' is used to
    support overlapping fields in SNANA, which is not currently
    supported here.
    """

    # known keywords
    KEY_FIELDID = 'LIBID'
    KEY_TELESCOPE = 'TELESCOPE'
    KEY_PIXSIZE = 'PIXSIZE'
    KEYS_FIELD = ['RA', 'DECL', 'MWEBV']
    KEYS_IGNORE = ['FIELD', 'NOBS', 'END_LIBID', 'END_OF_SIMLIB']

    header = OrderedDict()
    fields = OrderedDict()
    obs = OrderedDict()
    for col in ['field', 'search', 'date', 'idexpt', 'telescope', 'band',
                'pixsize', 'gain', 'noise', 'skysig', 'psf1', 'psf2',
                'psfratio', 'zptavg', 'zptsig', 'mag']:
        obs[col] = []

    # Initialize a few variables to None to indicated that they have not
    # yet been encountered when reading the file.
    fieldid = None
    telescope = None
    pixsize = None

    infile = open(filename, 'r')
    for line in infile.readlines():
        line = _stripcomment(line)  # Strip comments.
        words = line.split()  # Split on spaces.
        if len(words) == 0: continue

        # If the line is an observation...
        if words[0] in ['S:', 'T:']:

            # Check length of line.
            if len(words) < 13:
                raise ValueError("Improperly formatted observation line: '{}'"
                                 .format(line))
            
            # Check that necessary keywords were already encountered.
            if fieldid is None:
                raise ValueError("Observation line comes before LIBID keyword")
            if telescope is None:
                raise ValueError("Observation line comes before TELESCOPE "
                                 "keyword")
            if pixsize is None:
                raise ValueError("Observation line comes before PIXSIZE "
                                 "keyword")

            for key, val in [
                ('field', fieldid), ('search', words[0] == 'S:'),
                ('date', float(words[1])), ('idexpt', int(words[2])),
                ('telescope', telescope), ('band', words[3]),
                ('pixsize', pixsize), ('gain', float(words[4])),
                ('noise', float(words[5])), ('skysig', float(words[6])),
                ('psf1', float(words[7])), ('psf2', float(words[8])),
                ('psfratio', float(words[9])), ('zptavg', float(words[10])),
                ('zptsig', float(words[11])), ('mag', float(words[12]))]:
                obs[key].append(val)
    
        # Otherwise assume the line contains some keywords.
        else:
            # Put keyword, value pairs into a temporary dictionary.
            linevals = {}
            currentkw = None
            for word in words:
                if word[-1] == ':' and word[:-1].isupper():
                    currentkw = word[:-1]
                    linevals[currentkw] = None
                elif currentkw is None:
                    continue
                elif linevals[currentkw] is None:
                    linevals[currentkw] = word
                else:
                    linevals[currentkw] += ' ' + word

            # Do something with the keywords depending on what they are.
            for key, val in linevals.iteritems():
                if val is None: continue

                # Set field id and initialize new field
                if key == KEY_FIELDID:
                    fieldid = int(val)
                    fields[fieldid] = {}

                # Set current telescope or pixel scale
                elif key == KEY_TELESCOPE:
                    telescope = val
                elif key == KEY_PIXSIZE:
                    pixsize = float(val)

                # Set some values for the current field
                elif key in KEYS_FIELD:
                    if (fieldid is None) or (fieldid not in fields):
                        raise ValueError('Field keyword defined before LIBID')
                    fields[fieldid][key] = float(val)

                # Put any unidentified keywords into main header
                elif key not in KEYS_IGNORE:
                    header[key] = val

    infile.close()

    # Convert observations to a table
    obs = Table(obs)

    return header, fields, obs


def _stripcomment(line, char='#'):
    pos = line.find(char)
    if pos == -1: return line
    else: return line[:pos]
