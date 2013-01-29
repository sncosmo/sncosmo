# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Read and write functions for basic ascii tables with metadata."""

import os
from collections import OrderedDict
from astropy.io import ascii
from astropy.table import Table

__all__ = ['read', 'write']

def _stripcomment(line, char='#'):
    pos = line.find(char)
    if pos == -1: return line
    else: return line[:pos]

def read(filename, metachar='@', commentchar='#'):
    """Read a basic ASCII table with metadata.

    This can be deprecated once there is support for metadata in
    :mod:`astropy.io.ascii`.

    Parameters
    ----------
    filename : str or file object
    metachar : str, optional
        one-character string indicating a metadata line.
    commentchar : str, optional
        one-character string indicating comments.
    """
    meta = OrderedDict()

    if isinstance(filename, basestring):
        infile = open(filename, 'r')
    else:
        infile = filename

    # Read metadata.
    metalines = 0
    readingmeta = True
    rawline = infile.readline()
    while rawline != '' and readingmeta:
        line = _stripcomment(rawline)

        if len(line) == 0:  #line starts with comment or is blank
            rawline = infile.readline()
            continue

        line = line.strip()

        if len(line) == 0:  # Line is only whitespace
            metalines += 1
        elif line[0] == metachar: # line is metadata
            metalines += 1
            words = line[1:].split()

            if len(words) == 1:  # Just a keyword, no value
                meta[words[0]] = None

            elif len(words) > 1:  # Keyword and value(s).
                val = ' '.join(words[1:])
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        val = str(val)
            meta[words[0]] = val
                

        else:                     # Line is non-metadata, non-blank
            readingmeta = False
            infile.seek(-len(rawline), os.SEEK_CUR) #back up to start of line.
            continue

        rawline = infile.readline()

    # Read remaining lines using astropy.io.ascii
    table = Table.read(infile.readlines(), guess=False, format='ascii',
                       comment=commentchar)
    table.meta = meta
    return table


def write(table, filename, metachar='@', delimiter=' '):
    """Write a basic ASCII table with metadata.

    This can be deprecated once there is support for metadata in
    :mod:`astropy.io.ascii`.

    Parameters
    ----------
    table : astropy.table.Table
    filename : str or file object
    metachar : str, optional
        one-character string indicating a metadata line.
    """

    if isinstance(filename, basestring):
        outfile = open(filename, 'w')
    else:
        outfile = filename

    if table.meta is not None:
        for key, val in table.meta.iteritems():
            outfile.write('{}{} {}\n'.format(metachar, key, str(val)))
        outfile.write('\n')

    table.write(outfile, format='ascii', delimiter=' ')
