# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Importing this module registers loaders for built-in bandpasses.
# The module docstring, a table of the bandpasses, is generated at the
# after all the bandpasses are registered.

import string
from os.path import join

import numpy as np
from astropy.io import ascii
from astropy import units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.utils import OrderedDict
from astropy.config import ConfigurationItem
from astropy.extern import six

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
#  HST NICMOS, WFC3 and ACS bandpasses

def load_hst_bandpass(pkg_data_name, name=None):
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.AA, name=name)

# --------------------------------------------------------------------------
# HST NICMOS
nicmos_meta = {'filterset': 'nicmos2',
               'dataurl': 'http://www.stsci.edu/hst/',
               'retrieved': '05 Aug 2014',
               'description': 'Hubble Space Telescope NICMOS2 filters'}
registry.register_loader(Bandpass, 'nicf110w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_nicmos_nic2_f110w.dat'],
                         meta=nicmos_meta)
registry.register_loader(Bandpass, 'nicf160w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_nicmos_nic2_f160w.dat'],
                         meta=nicmos_meta)
del nicmos_meta

# --------------------------------------------------------------------------
# HST WFC3-IR
wfc3ir_meta = {'filterset': 'wfc3-ir',
               'dataurl': 'http://www.stsci.edu/hst/wfc3/ins_performance/'
                          'throughputs/Throughput_Tables',
               'retrieved': '05 Aug 2014',
               'description': 'Hubble Space Telescope WFC3 IR filters'}
registry.register_loader(Bandpass, 'f098m', load_hst_bandpass,
                         args=['../data/bandpasses/hst/hst_wfc3_ir_f098m.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f105w', load_hst_bandpass,
                         args=['../data/bandpasses/hst/hst_wfc3_ir_f105w.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f110w', load_hst_bandpass,
                         args=['../data/bandpasses/hst/hst_wfc3_ir_f110w.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f125w', load_hst_bandpass,
                         args=['../data/bandpasses/hst/hst_wfc3_ir_f125w.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f127m', load_hst_bandpass,
                         args=['../data/bandpasses/hst/hst_wfc3_ir_f127m.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f139m', load_hst_bandpass,
                         args=['../data/bandpasses/hst/hst_wfc3_ir_f139m.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f140w', load_hst_bandpass,
                         args=['../data/bandpasses/hst/hst_wfc3_ir_f140w.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f153m', load_hst_bandpass,
                         args=['../data/bandpasses/hst/hst_wfc3_ir_f153m.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f160w', load_hst_bandpass,
                         args=['../data/bandpasses/hst/hst_wfc3_ir_f160w.dat'],
                         meta=wfc3ir_meta)

del wfc3ir_meta

# --------------------------------------------------------------------------
# HST WFC3-UVIS
wfc3uvis_meta = {'filterset': 'wfc3-uvis',
                 'dataurl': 'http://www.stsci.edu/hst/wfc3/ins_performance/'
                            'throughputs/Throughput_Tables',
                 'retrieved': '05 Aug 2014',
                 'description': 'Hubble Space Telescope WFC3 UVIS filters'}
registry.register_loader(Bandpass, 'f218w',   load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f218w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f225w',   load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f225w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f275w',   load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f275w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f300x',   load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f300x.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f336w',   load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f336w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f350lp',  load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f350lp.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f390w',   load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f390w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f689m',   load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f689m.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f763m',   load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f763m.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f845m',   load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f845m.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f438w',   load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f438w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf475w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f475w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf555w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f555w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf606w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f606w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf625w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f625w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf775w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f775w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf814w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f814w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf850lp', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_wfc3_uvis_f850lp.dat'],
                         meta=wfc3uvis_meta)

del wfc3uvis_meta

# --------------------------------------------------------------------------
# HST ACS
acs_meta = {'filterset': 'acs',
            'dataurl': 'http://www.stsci.edu/hst/acs/analysis/throughputs',
            'retrieved': '05 Aug 2014',
            'description': 'Hubble Space Telescope ACS WFC filters'}
registry.register_loader(Bandpass, 'f435w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_acs_wfc_f435w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f475w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_acs_wfc_f475w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f555w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_acs_wfc_f555w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f606w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_acs_wfc_f606w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f625w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_acs_wfc_f625w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f775w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_acs_wfc_f775w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f814w', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_acs_wfc_f814w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f850lp', load_hst_bandpass, args=[
                         '../data/bandpasses/hst/hst_acs_wfc_f850lp.dat'],
                         meta=acs_meta)

del acs_meta


# --------------------------------------------------------------------------
#  JWST NIRCAM and MIRI bandpasses

def load_jwst_bandpass(pkg_data_name, name=None):
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.micron, name=name)

# --------------------------------------------------------------------------
# JWST NIRCAM Wide and medium bands
jwst_nircam_meta = {'filterset': 'jwst-nircam',
                    'dataurl': 'http://www.stsci.edu/jwst/instruments/nircam'
                               '/instrumentdesign/filters',
                    'retrieved': '09 Sep 2014',
                    'description': 'James Webb Space Telescope NIRCAM '
                    'Wide+Medium filters'}

registry.register_loader(Bandpass, 'f070w', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f070w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f090w', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f090w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f115w', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f115w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f150w', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f150w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f200w', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f200w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f277w', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f277w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f356w', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f356w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f444w', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f444w.dat'],
                         meta=jwst_nircam_meta)

registry.register_loader(Bandpass, 'f140m', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f140m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f162m', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f162m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f182m', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f182m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f210m', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f210m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f250m', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f250m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f300m', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f300m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f335m', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f335m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f360m', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f360m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f410m', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f410m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f430m', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f430m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f460m', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f460m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f480m', load_jwst_bandpass, args=[
                         '../data/bandpasses/jwst/jwst_nircam_f480m.dat'],
                         meta=jwst_nircam_meta)

del jwst_nircam_meta


# --------------------------------------------------------------------------
# JWST MIRI filters (idealized tophat functions)

def tophat_bandpass(ctr, width, name=None):
    """Create a tophat Bandpass centered at `ctr` with width `width` (both
    in microns. Sampling is fixed at 100 A == 0.01 microns"""

    wmin = ctr - width / 2. - 0.005
    wmax = ctr + width / 2. + 0.005
    wave = np.arange(wmin, wmax + 0.00001, 0.01)
    trans = np.ones_like(wave)
    trans[[0, -1]] = 0.
    return Bandpass(wave, trans, wave_unit=u.micron, name=name)

jwst_miri_meta = {'filterset': 'jwst-miri',
                  'dataurl': 'http://www.stsci.edu/jwst/instruments/miri/'
                             'instrumentdesign/filters',
                  'retrieved': '09 Sep 2014',
                  'description': 'James Webb Space Telescope MIRI '
                                 'filters (idealized tophats)'}

registry.register_loader(Bandpass, 'f560w', tophat_bandpass, args=[5.6, 1.2],
                         meta=jwst_miri_meta)
registry.register_loader(Bandpass, 'f770w', tophat_bandpass, args=[7.7, 2.2],
                         meta=jwst_miri_meta)
registry.register_loader(Bandpass, 'f1000w', tophat_bandpass, args=[10., 2.],
                         meta=jwst_miri_meta)
registry.register_loader(Bandpass, 'f1130w', tophat_bandpass, args=[11.3, 0.7],
                         meta=jwst_miri_meta)
registry.register_loader(Bandpass, 'f1280w', tophat_bandpass, args=[12.8, 2.4],
                         meta=jwst_miri_meta)
registry.register_loader(Bandpass, 'f1500w', tophat_bandpass, args=[15., 3.],
                         meta=jwst_miri_meta)
registry.register_loader(Bandpass, 'f1800w', tophat_bandpass, args=[18., 3.],
                         meta=jwst_miri_meta)
registry.register_loader(Bandpass, 'f2100w', tophat_bandpass, args=[21., 5.],
                         meta=jwst_miri_meta)
registry.register_loader(Bandpass, 'f2550w', tophat_bandpass, args=[25.5, 4.],
                         meta=jwst_miri_meta)
registry.register_loader(Bandpass, 'f1065c', tophat_bandpass,
                         args=[10.65, 0.53], meta=jwst_miri_meta)
registry.register_loader(Bandpass, 'f1140c', tophat_bandpass,
                         args=[11.4, 0.57], meta=jwst_miri_meta)
registry.register_loader(Bandpass, 'f1550c', tophat_bandpass,
                         args=[15.5, 0.78], meta=jwst_miri_meta)
registry.register_loader(Bandpass, 'f2300c', tophat_bandpass,
                         args=[23., 4.6], meta=jwst_miri_meta)

del jwst_miri_meta

# --------------------------------------------------------------------------
# Kepler filter

kepler_meta = {
    'filterset': 'kepler',
    'retrieved': '14 Jan 2015',
    'description': 'Bandpass for the Kepler spacecraft',
    'dataurl': 'http://keplergo.arc.nasa.gov/CalibrationResponse.shtml'}

registry.register_loader(Bandpass, 'kepler', load_bandpass,
                         args=['../data/bandpasses/kepler.dat'],
                         meta=kepler_meta)
del kepler_meta
