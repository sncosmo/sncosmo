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
                         dataurl=decam_url, retrieved=decam_retrieved)
registry.register_loader(Bandpass, 'desr', load_bandpass_ascii,
                         ['../data/bandpasses/des_r.dat'], filterset='des',
                         description='Dark Energy Camera r band',
                         dataurl=decam_url, retrieved=decam_retrieved)
registry.register_loader(Bandpass, 'desi', load_bandpass_ascii,
                         ['../data/bandpasses/des_i.dat'], filterset='des',
                         description='Dark Energy Camera i band',
                         dataurl=decam_url, retrieved=decam_retrieved)
registry.register_loader(Bandpass, 'desz', load_bandpass_ascii,
                         ['../data/bandpasses/des_z.dat'], filterset='des',
                         description='Dark Energy Camera z band',
                         dataurl=decam_url, retrieved=decam_retrieved)
registry.register_loader(Bandpass, 'desy', load_bandpass_ascii,
                         ['../data/bandpasses/des_y.dat'], filterset='des',
                         description='Dark Energy Camera y band',
                         dataurl=decam_url, retrieved=decam_retrieved)
del decam_url
del decam_retrieved

# --------------------------------------------------------------------------
# Bessel 1990

bessell_ref = (
    'B90', 'Bessell 1990 <http://adsabs.harvard.edu/abs/1990PASP..102.1181B>')
bessell_desc = 'Representation of Johnson-Cousins UBVRI system from Bessell (1990), Table 2'

registry.register_loader(
    Bandpass, 'bessellux', load_bandpass_ascii,
    ['../data/bandpasses/bessell_ux.dat'], filterset='bessell',
    description=bessell_desc, reference=bessell_ref)
registry.register_loader(
    Bandpass, 'bessellb', load_bandpass_ascii,
    ['../data/bandpasses/bessell_b.dat'], filterset='bessell',
    description=bessell_desc, reference=bessell_ref)
registry.register_loader(
    Bandpass, 'bessellv', load_bandpass_ascii,
    ['../data/bandpasses/bessell_v.dat'], filterset='bessell',
    description=bessell_desc, reference=bessell_ref)
registry.register_loader(
    Bandpass, 'bessellr', load_bandpass_ascii,
    ['../data/bandpasses/bessell_r.dat'], filterset='bessell',
    description=bessell_desc, reference=bessell_ref)
registry.register_loader(
    Bandpass, 'besselli', load_bandpass_ascii,
    ['../data/bandpasses/bessell_i.dat'], filterset='bessell',
    description=bessell_desc, reference=bessell_ref)

del bessell_ref
del bessell_desc

# --------------------------------------------------------------------------
# Generate docstring

lines = [
    '',
    '  '.join([11*'=', 80*'=', 14*'=', 8*'=', 12*'=']),
    '{:11}  {:80}  {:14}  {:8}  {:12}'
    .format('Name', 'Description', 'Reference', 'Data URL', 'Retrieved')
    ]
lines.append(lines[1])

urlnums = {}
allrefs = []
for m in registry.get_loaders_metadata(Bandpass):

    reflink = ''
    urllink = ''
    retrieved = ''

    if 'reference' in m:
        reflink = '[{}]_'.format(m['reference'][0])
        if m['reference'] not in allrefs:
            allrefs.append(m['reference'])

    if 'dataurl' in m:
        dataurl = m['dataurl']
        if dataurl not in urlnums:
            if len(urlnums) == 0: urlnums[dataurl] = 0
            else: urlnums[dataurl] = max(urlnums.values()) + 1
        urllink = '`{}`_'.format(string.letters[urlnums[dataurl]])

    if 'retrieved' in m:
        retrieved = m['retrieved']

    lines.append("{0!r:11}  {1:80}  {2:14}  {3:8}  {4:12}".format(
            m['name'], m['description'], reflink, urllink, retrieved))

lines.extend([lines[1], ''])
for refkey, ref in allrefs:
    lines.append('.. [{}] `{}`__'.format(refkey, ref))
lines.append('')
for url, urlnum in urlnums.iteritems():
    lines.append('.. _`{}`: {}'.format(string.letters[urlnum], url))
lines.append('')
__doc__ = '\n'.join(lines)

del lines
del urlnums
del allrefs
