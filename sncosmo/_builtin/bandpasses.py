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
des_meta = {'filterset': 'des',
            'dataurl': ('http://www.ctio.noao.edu/noao/content/'
                        'dark-energy-camera-decam'),
            'retrieved': '19 June 2012',
            'description': 'Dark Energy Camera grizy filter set'}
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
#  HST WFC3 and ACS bandpasses, added by S.Rodney

def load_hst_bandpass(pkg_data_name, name=None):
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.AA, name=name)

# --------------------------------------------------------------------------
# HST WFC3-IR
wfc3ir_meta = {'filterset': 'wfc3-ir',
            'dataurl': ('http://www.stsci.edu/hst/wfc3/ins_performance/throughputs/Throughput_Tables'
                        'wfc3ir'),
            'retrieved': '05 Aug 2014',
            'description': 'Hubble Space Telescope WFC3 IR filters'}
registry.register_loader(Bandpass, 'f098m', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_ir_f098m.dat'], meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f105w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_ir_f105w.dat'], meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f110w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_ir_f110w.dat'], meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f125w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_ir_f125w.dat'], meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f127m', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_ir_f127m.dat'], meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f139m', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_ir_f139m.dat'], meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f140w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_ir_f140w.dat'], meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f153m', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_ir_f153m.dat'], meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f160w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_ir_f160w.dat'], meta=wfc3ir_meta)

del wfc3ir_meta

# --------------------------------------------------------------------------
# HST WFC3-UVIS
wfc3uvis_meta = {'filterset': 'wfc3-uvis',
            'dataurl': ('http://www.stsci.edu/hst/wfc3/ins_performance/throughputs/Throughput_Tables'
                        'wfc3uvis'),
            'retrieved': '05 Aug 2014',
            'description': 'Hubble Space Telescope WFC3 UVIS filters'}
registry.register_loader(Bandpass, 'f218w',   load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f218w.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f225w',   load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f225w.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f275w',   load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f275w.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f300x',   load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f300x.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f336w',   load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f336w.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f350lp',  load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f350lp.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f390w',   load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f390w.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f689m',   load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f689m.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f763m',   load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f763m.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f845m',   load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f845m.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f438w',   load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f438w.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf475w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f475w.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf555w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f555w.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf606w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f606w.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf625w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f625w.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf775w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f775w.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf814w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f814w.dat'], meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf850lp',load_hst_bandpass, args=['../data/bandpasses/hst/hst_wfc3_uvis_f850lp.dat'], meta=wfc3uvis_meta)

del wfc3uvis_meta

# --------------------------------------------------------------------------
# HST ACS
acs_meta = {'filterset': 'acs',
            'dataurl': ('http://www.stsci.edu/hst/acs/analysis/throughputs'
                        'acs'),
            'retrieved': '05 Aug 2014',
            'description': 'Hubble Space Telescope ACS WFC filters'}
registry.register_loader(Bandpass, 'f435w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_acs_wfc_f435w.dat'], meta=acs_meta)
registry.register_loader(Bandpass, 'f475w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_acs_wfc_f475w.dat'], meta=acs_meta)
registry.register_loader(Bandpass, 'f555w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_acs_wfc_f555w.dat'], meta=acs_meta)
registry.register_loader(Bandpass, 'f606w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_acs_wfc_f606w.dat'], meta=acs_meta)
registry.register_loader(Bandpass, 'f625w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_acs_wfc_f625w.dat'], meta=acs_meta)
registry.register_loader(Bandpass, 'f775w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_acs_wfc_f775w.dat'], meta=acs_meta)
registry.register_loader(Bandpass, 'f814w', load_hst_bandpass, args=['../data/bandpasses/hst/hst_acs_wfc_f814w.dat'], meta=acs_meta)
registry.register_loader(Bandpass, 'f850lp',load_hst_bandpass, args=['../data/bandpasses/hst/hst_acs_wfc_f850lp.dat'],meta=acs_meta)

del acs_meta


# --------------------------------------------------------------------------
#  JWST NIRCAM and MIRI bandpasses, added by S.Rodney

def load_jwst_bandpass(pkg_data_name, name=None):
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.micron, name=name)

# --------------------------------------------------------------------------
# JWST NIRCAM Wide and medium bands
jwst_nircam_meta = {'filterset': 'jwst-nircam',
            'dataurl': ('http://www.stsci.edu/jwst/instruments/nircam/instrumentdesign/filters'
                        'jwstnircam'),
            'retrieved': '09 Sep 2014',
            'description': 'James Webb Space Telescope NIRCAM Wide+Medium filters'}

registry.register_loader(Bandpass, 'f070w', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f070w.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f090w', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f090w.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f115w', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f115w.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f150w', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f150w.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f200w', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f200w.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f277w', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f277w.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f356w', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f356w.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f444w', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f444w.dat'], meta=jwst_nircam_meta)

registry.register_loader(Bandpass, 'f140m', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f140m.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f162m', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f162m.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f182m', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f182m.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f210m', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f210m.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f250m', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f250m.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f300m', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f300m.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f335m', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f335m.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f360m', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f360m.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f410m', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f410m.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f430m', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f430m.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f460m', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f460m.dat'], meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f480m', load_jwst_bandpass, args=['../data/bandpasses/jwst/jwst_nircam_f480m.dat'], meta=jwst_nircam_meta)

del jwst_nircam_meta


# --------------------------------------------------------------------------
# JWST MIRI filters (idealized tophat functions)
jwst_miri_meta = {'filterset': 'jwst-miri',
            'dataurl': ('http://www.stsci.edu/jwst/instruments/miri/instrumentdesign/filters'
                        'jwstmiri'),
            'retrieved': '09 Sep 2014',
            'description': 'James Webb Space Telescope MIRI filters (idealized tophats)'}

import numpy as np
wave = np.arange( 1.0, 40.0, 0.05 )
tophat = lambda ctr,width : np.where( (ctr-width/2.<wave) & (wave<ctr+width/2.), 1, 0 )

registry.register( Bandpass(wave, tophat(5.6	, 1.2	), wave_unit=u.micron, name='f560w' ) )
registry.register( Bandpass(wave, tophat(7.7	, 2.2	), wave_unit=u.micron, name='f770w' ) )
registry.register( Bandpass(wave, tophat(10	, 2	), wave_unit=u.micron, name='f1000w' ) )
registry.register( Bandpass(wave, tophat(11.3	, 0.7	), wave_unit=u.micron, name='f1130w' ) )
registry.register( Bandpass(wave, tophat(12.8	, 2.4	), wave_unit=u.micron, name='f1280w' ) )
registry.register( Bandpass(wave, tophat(15	, 3	), wave_unit=u.micron, name='f1500w' ) )
registry.register( Bandpass(wave, tophat(18	, 3	), wave_unit=u.micron, name='f1800w' ) )
registry.register( Bandpass(wave, tophat(21	, 5	), wave_unit=u.micron, name='f2100w' ) )
registry.register( Bandpass(wave, tophat(25.5	, 4	), wave_unit=u.micron, name='f2550w' ) )
registry.register( Bandpass(wave, tophat(10.65	, 0.53	), wave_unit=u.micron, name='f1065c' ) )
registry.register( Bandpass(wave, tophat(11.4	, 0.57	), wave_unit=u.micron, name='f1140c' ) )
registry.register( Bandpass(wave, tophat(15.5	, 0.78	), wave_unit=u.micron, name='f1550c' ) )
registry.register( Bandpass(wave, tophat(23	, 4.6	), wave_unit=u.micron, name='f2300c' ) )

del jwst_miri_meta

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
