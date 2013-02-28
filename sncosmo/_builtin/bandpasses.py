# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Importing this module registers loaders for built-in bandpasses.
# The module docstring, a table of the bandpasses, is generated at the
# after all the bandpasses are registered.

import string
from os.path import join

from astropy.io import ascii
from astropy import units as u
from astropy.utils.data import (download_file, get_pkg_data_filename,
                                get_readable_fileobj)
from astropy.utils import OrderedDict
from astropy.config import ConfigurationItem

from .. import registry
from .. import Bandpass
from .. import utils

def load_bandpass_ascii(pkg_data_name):
    """Read two-column bandpass. First column is assumed to be wavelength
    in Angstroms."""
    
    filename = get_pkg_data_filename(pkg_data_name)
    t = ascii.read(filename, names=['disp', 'trans'])
    return Bandpass(t['disp'], t['trans'], dispersion_unit=u.AA)

# --------------------------------------------------------------------------
# DES

decam_url = 'http://www.ctio.noao.edu/noao/content/dark-energy-camera-decam'
decam_retrieved = '19 June 2012'
registry.register_loader(Bandpass, 'desg', load_bandpass_ascii,
                         ['../data/bandpasses/des_g.dat'], filterset='des',
                         description='Dark Energy Camera g band',
                         source=decam_url, retrieved=decam_retrieved)
registry.register_loader(Bandpass, 'desr', load_bandpass_ascii,
                         ['../data/bandpasses/des_r.dat'], filterset='des',
                         description='Dark Energy Camera r band',
                         source=decam_url, retrieved=decam_retrieved)
registry.register_loader(Bandpass, 'desi', load_bandpass_ascii,
                         ['../data/bandpasses/des_i.dat'], filterset='des',
                         description='Dark Energy Camera i band',
                         source=decam_url, retrieved=decam_retrieved)
registry.register_loader(Bandpass, 'desz', load_bandpass_ascii,
                         ['../data/bandpasses/des_z.dat'], filterset='des',
                         description='Dark Energy Camera z band',
                         source=decam_url, retrieved=decam_retrieved)
registry.register_loader(Bandpass, 'desy', load_bandpass_ascii,
                         ['../data/bandpasses/des_y.dat'], filterset='des',
                         description='Dark Energy Camera y band',
                         source=decam_url, retrieved=decam_retrieved)

del decam_url
del decam_retrieved

# --------------------------------------------------------------------------
# Generate docstring

lines = ['',
    '==========  =========================  ========  ====================',
    'Name        Description                Source    Retrieved',
    '==========  =========================  ========  ====================']
sourcenums = {}
for d in registry.get_loaders_metadata(Bandpass):
    if d['source'] in sourcenums:
        sourcenum = sourcenums[d['source']]
    else:
        if len(sourcenums) == 0:
            sourcenum = 0
        else:
            sourcenum = max(sourcenums.values()) + 1
        sourcenums[d['source']] = sourcenum

    sourceletter = string.letters[sourcenum]
    lines.append("{0:^10}  {1:^25}  {2:^8}  {3:20}".format(
            d['name'], d['description'], '`' + sourceletter + '`_', d['retrieved']))
lines.extend([lines[1], '', ''])

for source, sourcenum in sourcenums.iteritems():
    lines.append('.. _`{}`: {}'.format(string.letters[sourcenum], source))

__doc__ = '\n'.join(lines)

del lines
del sourcenums
