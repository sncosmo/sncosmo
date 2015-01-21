# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Importing this module registers loaders for built-in data structures:

- Sources
- Bandpasses
- MagSystems
"""

import string
import tarfile
import warnings
from os.path import join

import numpy as np
from astropy import wcs, units as u
from astropy.io import ascii, fits
from astropy.config import ConfigurationItem
from astropy.extern import six
from astropy.utils import OrderedDict
from astropy.utils.data import (download_file, get_pkg_data_filename,
                                get_readable_fileobj)

from . import registry
from . import io
from .models import Source, TimeSeriesSource, SALT2Source
from .spectral import (Bandpass, read_bandpass, Spectrum, MagSystem,
                       SpectralMagSystem, ABMagSystem)


# =============================================================================
# Bandpasses

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
                         args=['data/bandpasses/des_g.dat'], meta=des_meta)
registry.register_loader(Bandpass, 'desr', load_bandpass,
                         args=['data/bandpasses/des_r.dat'], meta=des_meta)
registry.register_loader(Bandpass, 'desi', load_bandpass,
                         args=['data/bandpasses/des_i.dat'], meta=des_meta)
registry.register_loader(Bandpass, 'desz', load_bandpass,
                         args=['data/bandpasses/des_z.dat'], meta=des_meta)
registry.register_loader(Bandpass, 'desy', load_bandpass,
                         args=['data/bandpasses/des_y.dat'], meta=des_meta)
del des_meta

# --------------------------------------------------------------------------
# Bessel 1990
bessell_meta = {
    'filterset': 'bessell',
    'reference': ('B90', '`Bessell 1990 <http://adsabs.harvard.edu/'
                  'abs/1990PASP..102.1181B>`__, Table 2'),
    'description': 'Representation of Johnson-Cousins UBVRI system'}

registry.register_loader(Bandpass, 'bessellux', load_bandpass,
                         args=['data/bandpasses/bessell_ux.dat'],
                         meta=bessell_meta)
registry.register_loader(Bandpass, 'bessellb', load_bandpass,
                         args=['data/bandpasses/bessell_b.dat'],
                         meta=bessell_meta)
registry.register_loader(Bandpass, 'bessellv', load_bandpass,
                         args=['data/bandpasses/bessell_v.dat'],
                         meta=bessell_meta)
registry.register_loader(Bandpass, 'bessellr', load_bandpass,
                         args=['data/bandpasses/bessell_r.dat'],
                         meta=bessell_meta)
registry.register_loader(Bandpass, 'besselli', load_bandpass,
                         args=['data/bandpasses/bessell_i.dat'],
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
                         args=['data/bandpasses/sdss_u.dat'],
                         meta=sdss_meta)
registry.register_loader(Bandpass, 'sdssg', load_bandpass,
                         args=['data/bandpasses/sdss_g.dat'],
                         meta=sdss_meta)
registry.register_loader(Bandpass, 'sdssr', load_bandpass,
                         args=['data/bandpasses/sdss_r.dat'],
                         meta=sdss_meta)
registry.register_loader(Bandpass, 'sdssi', load_bandpass,
                         args=['data/bandpasses/sdss_i.dat'],
                         meta=sdss_meta)
registry.register_loader(Bandpass, 'sdssz', load_bandpass,
                         args=['data/bandpasses/sdss_z.dat'],
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
                         'data/bandpasses/hst/hst_nicmos_nic2_f110w.dat'],
                         meta=nicmos_meta)
registry.register_loader(Bandpass, 'nicf160w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_nicmos_nic2_f160w.dat'],
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
                         args=['data/bandpasses/hst/hst_wfc3_ir_f098m.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f105w', load_hst_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f105w.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f110w', load_hst_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f110w.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f125w', load_hst_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f125w.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f127m', load_hst_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f127m.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f139m', load_hst_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f139m.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f140w', load_hst_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f140w.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f153m', load_hst_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f153m.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f160w', load_hst_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f160w.dat'],
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
                         'data/bandpasses/hst/hst_wfc3_uvis_f218w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f225w',   load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f225w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f275w',   load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f275w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f300x',   load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f300x.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f336w',   load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f336w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f350lp',  load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f350lp.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f390w',   load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f390w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f689m',   load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f689m.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f763m',   load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f763m.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f845m',   load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f845m.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f438w',   load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f438w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf475w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f475w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf555w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f555w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf606w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f606w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf625w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f625w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf775w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f775w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf814w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f814w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf850lp', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f850lp.dat'],
                         meta=wfc3uvis_meta)

del wfc3uvis_meta

# --------------------------------------------------------------------------
# HST ACS
acs_meta = {'filterset': 'acs',
            'dataurl': 'http://www.stsci.edu/hst/acs/analysis/throughputs',
            'retrieved': '05 Aug 2014',
            'description': 'Hubble Space Telescope ACS WFC filters'}
registry.register_loader(Bandpass, 'f435w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f435w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f475w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f475w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f555w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f555w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f606w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f606w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f625w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f625w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f775w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f775w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f814w', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f814w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f850lp', load_hst_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f850lp.dat'],
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
                         'data/bandpasses/jwst/jwst_nircam_f070w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f090w', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f090w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f115w', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f115w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f150w', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f150w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f200w', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f200w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f277w', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f277w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f356w', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f356w.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f444w', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f444w.dat'],
                         meta=jwst_nircam_meta)

registry.register_loader(Bandpass, 'f140m', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f140m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f162m', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f162m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f182m', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f182m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f210m', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f210m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f250m', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f250m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f300m', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f300m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f335m', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f335m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f360m', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f360m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f410m', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f410m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f430m', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f430m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f460m', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f460m.dat'],
                         meta=jwst_nircam_meta)
registry.register_loader(Bandpass, 'f480m', load_jwst_bandpass, args=[
                         'data/bandpasses/jwst/jwst_nircam_f480m.dat'],
                         meta=jwst_nircam_meta)

del jwst_nircam_meta


# --------------------------------------------------------------------------
# JWST MIRI filters (idealized tophat functions)

def tophat_bandpass(ctr, width, name=None):
    """Create a tophat Bandpass centered at `ctr` with width `width` (both
    in microns. Sampling is fixed at 100 A == 0.01 microns"""

    nintervals = 100  # intervals between wmin and wmax
    interval = width / nintervals  # interval of each sample
    wmin = ctr - width / 2. - interval / 2.
    wmax = ctr + width / 2. + interval / 2.
    wave = np.linspace(wmin, wmax, nintervals + 1)
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
                         args=['data/bandpasses/kepler.dat'],
                         meta=kepler_meta)
del kepler_meta


# =============================================================================
# Sources

def load_timeseries_ascii(remote_url, name=None, version=None):
    with get_readable_fileobj(remote_url, cache=True) as f:
        phases, wavelengths, flux = io.read_griddata_ascii(f)
    return TimeSeriesSource(phases, wavelengths, flux,
                            name=name, version=version)


def load_timeseries_fits(remote_url, name=None, version=None):
    fn = download_file(remote_url, cache=True)
    phase, wave, flux = io.read_griddata_fits(fn)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version)


def load_timeseries_fits_local(pkg_data_name, name=None, version=None):
    fname = get_pkg_data_filename(pkg_data_name)
    phase, wave, flux = io.read_griddata_fits(fname)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version)


# ------------------------------------------------------------------------
# Nugent models

baseurl = 'https://c3.lbl.gov/nugent/templates/'
website = 'https://c3.lbl.gov/nugent/nugent_templates.html'
subclass = '`~sncosmo.TimeSeriesSource`'
n02ref = ('N02', 'Nugent, Kim & Permutter 2002 '
          '<http://adsabs.harvard.edu/abs/2002PASP..114..803N>')
s04ref = ('S04', 'Stern, et al. 2004 '
          '<http://adsabs.harvard.edu/abs/2004ApJ...612..690S>')
l05ref = ('L05', 'Levan et al. 2005 '
          '<http://adsabs.harvard.edu/abs/2005ApJ...624..880L>')
g99ref = ('G99', 'Gilliland, Nugent & Phillips 1999 '
          '<http://adsabs.harvard.edu/abs/1999ApJ...521...30G>')

registry.register_loader(Source, 'nugent-sn1a', load_timeseries_ascii,
                         args=[baseurl + 'sn1a_flux.v1.2.dat.gz'],
                         version='1.2',
                         meta={'url': website, 'type': 'SN Ia',
                               'subclass': subclass, 'reference': n02ref})
registry.register_loader(Source, 'nugent-sn91t', load_timeseries_ascii,
                         args=[baseurl + 'sn91t_flux.v1.1.dat.gz'],
                         version='1.1',
                         meta={'url': website, 'type': 'SN Ia',
                               'subclass': subclass, 'reference': s04ref})
registry.register_loader(Source, 'nugent-sn91bg', load_timeseries_ascii,
                         args=[baseurl + 'sn91bg_flux.v1.1.dat.gz'],
                         version='1.1',
                         meta={'url': website, 'type': 'SN Ia',
                               'subclass': subclass, 'reference': n02ref})
registry.register_loader(Source, 'nugent-sn1bc', load_timeseries_ascii,
                         args=[baseurl + 'sn1bc_flux.v1.1.dat.gz'],
                         version='1.1',
                         meta={'url': website, 'type': 'SN Ib/c',
                               'subclass': subclass, 'reference': l05ref})
registry.register_loader(Source, 'nugent-hyper', load_timeseries_ascii,
                         args=[baseurl + 'hyper_flux.v1.2.dat.gz'],
                         version='1.2',
                         meta={'url': website, 'type': 'SN Ib/c',
                               'subclass': subclass, 'reference': l05ref})
registry.register_loader(Source, 'nugent-sn2p', load_timeseries_ascii,
                         args=[baseurl + 'sn2p_flux.v1.2.dat.gz'],
                         version='1.2',
                         meta={'url': website, 'type': 'SN IIP',
                               'subclass': subclass, 'reference': g99ref})
registry.register_loader(Source, 'nugent-sn2l', load_timeseries_ascii,
                         args=[baseurl + 'sn2l_flux.v1.2.dat.gz'],
                         version='1.2',
                         meta={'url': website, 'type': 'SN IIL',
                               'subclass': subclass, 'reference': g99ref})
registry.register_loader(Source, 'nugent-sn2n', load_timeseries_ascii,
                         args=[baseurl + 'sn2n_flux.v2.1.dat.gz'],
                         version='2.1',
                         meta={'url': website, 'type': 'SN IIn',
                               'subclass': subclass, 'reference': g99ref})


# -----------------------------------------------------------------------
# Sako et al 2011 models

baseurl = 'http://sncosmo.github.io/data/models/'
ref = ('S11', 'Sako et al. 2011 '
       '<http://adsabs.harvard.edu/abs/2011ApJ...738..162S>')
website = 'http://sdssdp62.fnal.gov/sdsssn/SNANA-PUBLIC/'
subclass = '`~sncosmo.TimeSeriesSource`'
note = "extracted from SNANA's SNDATA_ROOT on 29 March 2013."

registry.register_loader(Source, 's11-2004hx', load_timeseries_ascii,
                         args=[baseurl + 'S11_SDSS-000018.SED'], version='1.0',
                         meta={'url': website, 'type': 'SN IIL/P',
                               'subclass': subclass, 'reference': ref,
                               'note': note})
registry.register_loader(Source, 's11-2005lc', load_timeseries_ascii,
                         args=[baseurl + 'S11_SDSS-001472.SED'], version='1.0',
                         meta={'url': website, 'type': 'SN IIP',
                               'subclass': subclass, 'reference': ref,
                               'note': note})
registry.register_loader(Source, 's11-2005hl', load_timeseries_ascii,
                         args=[baseurl + 'S11_SDSS-002000.SED'], version='1.0',
                         meta={'url': website, 'type': 'SN Ib',
                               'subclass': subclass, 'reference': ref,
                               'note': note})
registry.register_loader(Source, 's11-2005hm', load_timeseries_ascii,
                         args=[baseurl + 'S11_SDSS-002744.SED'], version='1.0',
                         meta={'url': website, 'type': 'SN Ib',
                               'subclass': subclass, 'reference': ref,
                               'note': note})
registry.register_loader(Source, 's11-2005gi', load_timeseries_ascii,
                         args=[baseurl + 'S11_SDSS-003818.SED'], version='1.0',
                         meta={'url': website, 'type': 'SN IIP',
                               'subclass': subclass, 'reference': ref,
                               'note': note})
registry.register_loader(Source, 's11-2006fo', load_timeseries_ascii,
                         args=[baseurl + 'S11_SDSS-013195.SED'], version='1.0',
                         meta={'url': website, 'type': 'SN Ic',
                               'subclass': subclass, 'reference': ref,
                               'note': note})
registry.register_loader(Source, 's11-2006jo', load_timeseries_ascii,
                         args=[baseurl + 'S11_SDSS-014492.SED'], version='1.0',
                         meta={'url': website, 'type': 'SN Ib',
                               'subclass': subclass, 'reference': ref,
                               'note': note})
registry.register_loader(Source, 's11-2006jl', load_timeseries_ascii,
                         args=[baseurl + 'S11_SDSS-014599.SED'], version='1.0',
                         meta={'url': website, 'type': 'SN IIP',
                               'subclass': subclass, 'reference': ref,
                               'note': note})

# -----------------------------------------------------------------------
# Hsiao models

baseurl = 'http://sncosmo.github.io/data/models/'
hsiao_meta = {'url': 'http://csp.obs.carnegiescience.edu/data/snpy',
              'type': 'SN Ia',
              'subclass': '`~sncosmo.TimeSeriesSource`',
              'reference': ('H07', 'Hsiao et al. 2007 '
                            '<http://adsabs.harvard.edu/abs/'
                            '2007ApJ...663.1187H>'),
              'note': 'extracted from the SNooPy package on 21 Dec 2012.'}
registry.register_loader(Source, 'hsiao', load_timeseries_fits,
                         args=[baseurl + 'Hsiao_SED.fits'], version='1.0',
                         meta=hsiao_meta)
registry.register_loader(Source, 'hsiao', load_timeseries_fits,
                         args=[baseurl + 'Hsiao_SED_V2.fits'], version='2.0',
                         meta=hsiao_meta)
registry.register_loader(Source, 'hsiao', load_timeseries_fits,
                         args=[baseurl + 'Hsiao_SED_V3.fits'], version='3.0',
                         meta=hsiao_meta)


# -------------------------------------------------------------------------
# Hsiao subsampled

meta = {'url': 'http://csp.obs.carnegiescience.edu/data/snpy',
        'type': 'SN Ia',
        'subclass': '`~sncosmo.TimeSeriesSource`',
        'reference': ('H07',
                      'Hsiao et al. 2007 <http://adsabs.harvard.edu/abs/'
                      '2007ApJ...663.1187H>'),
        'note': 'extracted from the SNooPy package on 21 Dec 2012.'}
registry.register_loader(Source, 'hsiao-subsampled',
                         load_timeseries_fits_local,
                         args=['data/models/Hsiao_SED_V3_subsampled.fits'],
                         version='3.0', meta=meta)


# -----------------------------------------------------------------------
# SALT2 models

def load_salt2model(remote_url, topdir, name=None, version=None):
    fn = download_file(remote_url, cache=True)
    t = tarfile.open(fn, 'r:gz')

    errscalefn = join(topdir, 'salt2_spec_dispersion_scaling.dat')
    if errscalefn in t.getnames():
        errscalefile = t.extractfile(errscalefn)
    else:
        errscalefile = None

    model = SALT2Source(
        m0file=t.extractfile(join(topdir, 'salt2_template_0.dat')),
        m1file=t.extractfile(join(topdir, 'salt2_template_1.dat')),
        clfile=t.extractfile(join(topdir, 'salt2_color_correction.dat')),
        cdfile=t.extractfile(join(topdir, 'salt2_color_dispersion.dat')),
        errscalefile=t.extractfile(
            join(topdir, 'salt2_lc_dispersion_scaling.dat')),
        lcrv00file=t.extractfile(
            join(topdir, 'salt2_lc_relative_variance_0.dat')),
        lcrv11file=t.extractfile(
            join(topdir, 'salt2_lc_relative_variance_1.dat')),
        lcrv01file=t.extractfile(
            join(topdir, 'salt2_lc_relative_covariance_01.dat')),
        name=name,
        version=version)

    t.close()

    return model

baseurl = 'http://supernovae.in2p3.fr/salt/lib/exe/fetch.php?media='
website = 'http://supernovae.in2p3.fr/salt/doku.php?id=salt_templates'
g07ref = ('G07', 'Guy et al. 2007 '
          '<http://adsabs.harvard.edu/abs/2007A%26A...466...11G>')
g10ref = ('G10', 'Guy et al. 2010 '
          '<http://adsabs.harvard.edu/abs/2010A%26A...523A...7G>')
b14ref = ('B14', 'Betoule et al. 2014 '
          '<http://arxiv.org/abs/1401.4064>')
registry.register_loader(
    Source, 'salt2', load_salt2model,
    args=[baseurl + 'salt2_model_data-1-0.tar.gz', 'salt2'],
    version='1.0',
    meta={'type': 'SN Ia', 'subclass': '`~sncosmo.SALT2Source`',
          'url': website, 'reference': g07ref})
registry.register_loader(
    Source, 'salt2', load_salt2model,
    args=[baseurl + 'salt2_model_data-1-1.tar.gz', 'salt2-1-1'],
    version='1.1',
    meta={'type': 'SN Ia', 'subclass': '`~sncosmo.SALT2Source`',
          'url': website, 'reference': g07ref})
registry.register_loader(
    Source, 'salt2', load_salt2model,
    args=[baseurl + 'salt2_model_data-2-0.tar.gz', 'salt2-2-0'],
    version='2.0',
    meta={'type': 'SN Ia', 'subclass': '`~sncosmo.SALT2Source`',
          'url': website, 'reference': g10ref})
registry.register_loader(
    Source, 'salt2', load_salt2model,
    args=[baseurl + 'salt2_model_data-2-4.tar.gz', 'salt2-4'],
    version='2.4',
    meta={'type': 'SN Ia', 'subclass': '`~sncosmo.SALT2Source`',
          'url': website, 'reference': b14ref})


# --------------------------------------------------------------------------
# SALT2 extended

registry.register_loader(
    Source, 'salt2-extended', load_salt2model,
    args=['http://sncosmo.github.io/data/models/salt2_extended.tar.gz',
          'salt2_extended'],
    version='1.0',
    meta={'type': 'SN Ia',
          'subclass': '`~sncosmo.SALT2Source`',
          'url': 'http://sdssdp62.fnal.gov/sdsssn/SNANA-PUBLIC/',
          'note': "extracted from SNANA's SNDATA_ROOT on 15 August 2013."})


# --------------------------------------------------------------------------
# 2011fe

def load_2011fe(name=None, version=None):

    remote_url = "http://snfactory.lbl.gov/snf/data/SN2011fe.tar.gz"

    # filter warnings about RADESYS keyword in files
    warnings.filterwarnings('ignore', category=wcs.FITSFixedWarning,
                            append=True)

    tarfname = download_file(remote_url, cache=True)
    t = tarfile.open(tarfname, 'r:gz')
    phasestrs = []
    spectra = []
    disp = None
    for fname in t.getnames():
        if fname[-4:] == '.fit':
            hdulist = fits.open(t.extractfile(fname))
            flux_density = hdulist[0].data
            phasestrs.append(fname[-8:-4])  # like 'P167' or 'M167'
            spectra.append(flux_density)

            # Get dispersion values if we haven't yet
            # (dispersion should be the same for all)
            if disp is None:
                w = wcs.WCS(hdulist[0].header)
                nflux = len(flux_density)
                idx = np.arange(nflux)  # pixel coords
                idx.shape = (nflux, 1)  # make it 2-d
                disp = w.wcs_pix2world(idx, 0)[:, 0]

            hdulist.close()

    # get phases in floats
    phases = []
    for phasestr in phasestrs:
        phase = 0.1 * float(phasestr[1:])
        if phasestr[0] == 'M':
            phase = -phase
        phases.append(phase)

    # Add a point at explosion.
    # The phase of explosion is given in the paper as
    # t_expl - t_bmax = 55796.696 - 55814.51 = -17.814
    # where t_expl is from Nugent et al (2012)
    phases.append(-17.814)
    spectra.append(np.zeros_like(spectra[0]))

    # order spectra and put them all together
    spectra = sorted(zip(phases, spectra), key=lambda x: x[0])
    flux = np.array([s[1] for s in spectra])

    phases = np.array(phases)
    phases.sort()

    return TimeSeriesSource(phases, disp, flux,
                            name=name, version=version)


p13ref = ('P13', 'Pereira et al. 2013 '
          '<http://adsabs.harvard.edu/abs/2013A%26A...554A..27P>')
registry.register_loader(Source, '2011fe', load_2011fe, version='1.0',
                         meta={'type': 'SN Ia',
                               'subclass': '`~sncosmo.TimeSeriesSource`',
                               'url': 'http://snfactory.lbl.gov/snf/data',
                               'reference': p13ref})


# --------------------------------------------------------------------------
# SNANA CC SN models

baseurl = 'http://sncosmo.github.io/data/models/snana/'
meta = {'url': 'http://das.sdss2.org/ge/sample/sdsssn/SNANA-PUBLIC/',
        'type': 'Ic',
        'subclass': '`~sncosmo.TimeSeriesSource`',
        'reference': ('SNANA', 'Kessler et al. 2009 '
                      '<http://adsabs.harvard.edu/abs/2009PASP..121.1028K>'),
        'note': "extracted from SNANA's SNDATA_ROOT on 5 August 2014."}

for modelname, sedfile, type in [
    ['Ic.01', 'CSP-2004fe.SED',  'Ic'],
    ['Ic.02', 'CSP-2004gq.SED',  'Ic'],
    ['Ic.03', 'SDSS-004012.SED', 'Ic'],
    ['Ic.04', 'SDSS-013195.SED', 'Ic'],  # PSNID
    ['Ic.05', 'SDSS-014475.SED', 'Ic'],
    ['Ic.06', 'SDSS-015475.SED', 'Ic'],
    ['Ic.07', 'SDSS-017548.SED', 'Ic'],
    ['Ic.08', 'SNLS-04D1la.SED', 'Ic'],
    ['Ic.09', 'SNLS-04D4jv.SED', 'Ic'],
    ['Ib.01', 'CSP-2004gv.SED',  'Ib'],
    ['Ib.02', 'CSP-2006ep.SED',  'Ib'],
    ['Ib.03', 'CSP-2007Y.SED',   'Ib'],
    ['Ib.04', 'SDSS-000020.SED', 'Ib'],
    ['Ib.05', 'SDSS-002744.SED', 'Ib'],  # PSNID
    ['Ib.06', 'SDSS-014492.SED', 'Ib'],  # PSNID
    ['Ib.07', 'SDSS-019323.SED', 'Ib'],
    ['IIP.01', 'SDSS-000018.SED', 'IIP'],  # PSNID
    ['IIP.02', 'SDSS-003818.SED', 'IIP'],  # PSNID
    ['IIP.03', 'SDSS-013376.SED', 'IIP'],
    ['IIP.04', 'SDSS-014450.SED', 'IIP'],
    ['IIP.05', 'SDSS-014599.SED', 'IIP'],  # PSNID
    ['IIP.06', 'SDSS-015031.SED', 'IIP'],
    ['IIP.07', 'SDSS-015320.SED', 'IIP'],
    ['IIP.08', 'SDSS-015339.SED', 'IIP'],
    ['IIP.09', 'SDSS-017564.SED', 'IIP'],
    ['IIP.10', 'SDSS-017862.SED', 'IIP'],
    ['IIP.11', 'SDSS-018109.SED', 'IIP'],
    ['IIP.12', 'SDSS-018297.SED', 'IIP'],
    ['IIP.13', 'SDSS-018408.SED', 'IIP'],
    ['IIP.14', 'SDSS-018441.SED', 'IIP'],
    ['IIP.15', 'SDSS-018457.SED', 'IIP'],
    ['IIP.16', 'SDSS-018590.SED', 'IIP'],
    ['IIP.17', 'SDSS-018596.SED', 'IIP'],
    ['IIP.18', 'SDSS-018700.SED', 'IIP'],
    ['IIP.19', 'SDSS-018713.SED', 'IIP'],
    ['IIP.20', 'SDSS-018734.SED', 'IIP'],
    ['IIP.21', 'SDSS-018793.SED', 'IIP'],
    ['IIP.22', 'SDSS-018834.SED', 'IIP'],
    ['IIP.23', 'SDSS-018892.SED', 'IIP'],
    ['IIP.24', 'SDSS-020038.SED', 'IIP'],
    ['IIn.01', 'SDSS-012842.SED', 'IIN'],
    ['IIn.02', 'SDSS-013449.SED', 'IIN'],
]:

    meta.update({'snid': sedfile.split('.')[0], 'type': type})
    registry.register_loader(Source, modelname, load_timeseries_ascii,
                             args=[baseurl + sedfile], version='1.0',
                             meta=meta)

# --------------------------------------------------------------------------
# Pop III CC SN models from D.Whalen et al. 2013.

baseurl = 'http://sncosmo.github.io/data/models/whalen/'
meta = {'snid': 'z15B', 'type': 'PopIII',
        'subclass': '`~sncosmo.TimeSeriesSource`',
        'reference': ('Whalen13',
                      'Whalen et al. 2013 '
                      '<http://adsabs.harvard.edu/abs/2013ApJ...768...95W>'),
        'note': "private communication (D.Whalen, May 2014)."}

for mod in ['z15B', 'z15D', 'z15G', 'z25B', 'z25D', 'z25G', 'z40B', 'z40G']:
    meta.update({'snid': mod})
    datfile = 'popIII-%s.sed.restframe10pc.dat' % mod
    registry.register_loader(Source, mod, load_timeseries_ascii,
                             args=[baseurl + datfile], version='1.0',
                             meta=meta)

# Clean up the module namespace.
del n02ref
del s04ref
del l05ref
del g99ref
del g07ref
del g10ref
del p13ref
del baseurl
del ref
del website
del subclass
del note


# =============================================================================
# MagSystems

# ---------------------------------------------------------------------------
# AB system
def load_ab(name=None):
    return ABMagSystem(name=name)
registry.register_loader(
    MagSystem, 'ab', load_ab,
    meta={'subclass': '`~sncosmo.ABMagSystem`',
          'description': 'Source of 3631 Jy has magnitude 0 in all bands'})


# ---------------------------------------------------------------------------
# Spectral systems
def load_spectral_magsys_fits(remote_url, name=None):
    fn = download_file(remote_url, cache=True)
    hdulist = fits.open(fn)
    dispersion = hdulist[1].data['WAVELENGTH']
    flux_density = hdulist[1].data['FLUX']
    hdulist.close()
    refspectrum = Spectrum(dispersion, flux_density,
                           unit=(u.erg / u.s / u.cm**2 / u.AA), wave_unit=u.AA)
    return SpectralMagSystem(refspectrum, name=name)

calspec_url = 'ftp://ftp.stsci.edu/cdbs/calspec/'

registry.register_loader(
    MagSystem, 'vega', load_spectral_magsys_fits,
    args=[calspec_url + 'alpha_lyr_stis_007.fits'],
    meta={'subclass': '`~sncosmo.SpectralMagSystem`', 'url': calspec_url,
          'description': 'Vega (alpha lyrae) has magnitude 0 in all bands'})

registry.register_loader(
    MagSystem, 'bd17', load_spectral_magsys_fits,
    args=[calspec_url + 'bd_17d4708_stisnic_005.fits'],
    meta={'subclass': '`~sncosmo.SpectralMagSystem`', 'url': calspec_url,
          'description': 'BD+17d4708 has magnitude 0 in all bands.'})

del calspec_url
