# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Importing this module registers loaders for built-in bandpasses.
# The module docstring, a table of the bandpasses, is generated at the
# after all the bandpasses are registered.

import string
from os.path import join

from astropy.io import ascii
from astropy import units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.utils import OrderedDict
from astropy.config import ConfigurationItem

from .. import registry
from ..spectral import Bandpass, read_bandpass


def load_bandpass(pkg_data_name, name=None):
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.AA, name=name)

# --------------------------------------------------------------------------
# DES
des_meta = {
    'filterset': 'des',
    'retrieved': '22 March 2013',
    'description': 'Dark Energy Camera grizy filter set at airmass 1.3'}
registry.register_loader(Bandpass, 'desg', load_bandpass,
                         args=['../data/bandpasses/des_g.dat'], meta=des_meta)
registry.register_loader(Bandpass, 'desr', load_bandpass,
                         args=['../data/bandpasses/des_r.dat'], meta=des_meta)
registry.register_loader(Bandpass, 'desi', load_bandpass,
                         args=['../data/bandpasses/des_i.dat'], meta=des_meta)
registry.register_loader(Bandpass, 'desz', load_bandpass,
                         args=['../data/bandpasses/des_z.dat'], meta=des_meta)
registry.register_loader(Bandpass, 'desy', load_bandpass,
                         args=['../data/bandpasses/des_y.dat'], meta=des_meta)
del des_meta

# --------------------------------------------------------------------------
# Bessel 1990
bessell_meta = {
    'filterset': 'bessell',
    'reference': ('B90', '`Bessell 1990 <http://adsabs.harvard.edu/'
                  'abs/1990PASP..102.1181B>`__, Table 2'),
    'description': 'Representation of Johnson-Cousins UBVRI system'}

registry.register_loader(Bandpass, 'bessellux', load_bandpass,
                         args=['../data/bandpasses/bessell_ux.dat'],
                         meta=bessell_meta)
registry.register_loader(Bandpass, 'bessellb', load_bandpass,
                         args=['../data/bandpasses/bessell_b.dat'],
                         meta=bessell_meta)
registry.register_loader(Bandpass, 'bessellv', load_bandpass,
                         args=['../data/bandpasses/bessell_v.dat'],
                         meta=bessell_meta)
registry.register_loader(Bandpass, 'bessellr', load_bandpass,
                         args=['../data/bandpasses/bessell_r.dat'],
                         meta=bessell_meta)
registry.register_loader(Bandpass, 'besselli', load_bandpass,
                         args=['../data/bandpasses/bessell_i.dat'],
                         meta=bessell_meta)
del bessell_meta

# --------------------------------------------------------------------------
# SDSS
sdss_meta = {
    'filterset': 'sdss',
    'reference': ('D10', '`Doi et al. 2010 <http://adsabs.harvard.edu/'
                  'abs/2010AJ....139.1628D>`__, Table 4'),
    'description':
        'SDSS 2.5m imager at airmass 1.3 (including atmosphere), normalized'}

registry.register_loader(Bandpass, 'sdssu', load_bandpass,
                         args=['../data/bandpasses/sdss_u.dat'],
                         meta=sdss_meta)
registry.register_loader(Bandpass, 'sdssg', load_bandpass,
                         args=['../data/bandpasses/sdss_g.dat'],
                         meta=sdss_meta)
registry.register_loader(Bandpass, 'sdssr', load_bandpass,
                         args=['../data/bandpasses/sdss_r.dat'],
                         meta=sdss_meta)
registry.register_loader(Bandpass, 'sdssi', load_bandpass,
                         args=['../data/bandpasses/sdss_i.dat'],
                         meta=sdss_meta)
registry.register_loader(Bandpass, 'sdssz', load_bandpass,
                         args=['../data/bandpasses/sdss_z.dat'],
                         meta=sdss_meta)
del sdss_meta

# --------------------------------------------------------------------------
# Generate docstring

lines = [
    '',
    '  '.join([11*'=', 80*'=', 14*'=', 8*'=', 12*'=']),
    '{0:11}  {1:80}  {2:14}  {3:8}  {4:12}'
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
        reflink = '[{0}]_'.format(m['reference'][0])
        if m['reference'] not in allrefs:
            allrefs.append(m['reference'])

    if 'dataurl' in m:
        dataurl = m['dataurl']
        if dataurl not in urlnums:
            if len(urlnums) == 0:
                urlnums[dataurl] = 0
            else:
                urlnums[dataurl] = max(urlnums.values()) + 1
        urllink = '`{0}`_'.format(string.letters[urlnums[dataurl]])

    if 'retrieved' in m:
        retrieved = m['retrieved']

    lines.append("{0!r:11}  {1:80}  {2:14}  {3:8}  {4:12}".format(
        m['name'], m['description'], reflink, urllink, retrieved))

lines.extend([lines[1], ''])
for refkey, ref in allrefs:
    lines.append('.. [{0}] {1}'.format(refkey, ref))
lines.append('')
for url, urlnum in urlnums.iteritems():
    lines.append('.. _`{0}`: {1}'.format(string.letters[urlnum], url))
lines.append('')
__doc__ = '\n'.join(lines)

del lines
del urlnums
del allrefs
