# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Importing this module registers loaders for built-in data structures:

- Sources
- Bandpasses
- MagSystems
"""

import os
import warnings
from os.path import join

import numpy as np
from astropy import units as u, wcs
from astropy.config import get_cache_dir
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

from . import conf
from . import io
from . import snfitio
from .bandpasses import (
    Bandpass, _BANDPASSES, _BANDPASS_INTERPOLATORS, read_bandpass)

from .constants import BANDPASS_TRIM_LEVEL
from .magsystems import (
    ABMagSystem, CompositeMagSystem, SpectralMagSystem, _MAGSYSTEMS)

from .models import (
    MLCS2k2Source, SALT2Source, SALT3Source, SNEMOSource, SUGARSource,
    TimeSeriesSource, _SOURCES)

from .specmodel import SpectrumModel
from .utils import DataMirror

# This module is only imported for its side effects.
__all__ = []


def get_rootdir():
    # use the environment variable if set
    data_dir = os.environ.get('SNCOSMO_DATA_DIR')

    # otherwise, use config file value if set.
    if data_dir is None:
        data_dir = conf.data_dir

    # if still None, use astropy cache dir (and create if necessary!)
    if data_dir is None:
        data_dir = join(get_cache_dir(), "sncosmo")
        if not os.path.isdir(data_dir):
            if os.path.exists(data_dir):
                raise RuntimeError("{0} not a directory".format(data_dir))
            os.mkdir(data_dir)

    return data_dir


DATADIR = DataMirror(get_rootdir, "http://sncosmo.github.io/data")


# =============================================================================
# Bandpasses


def load_bandpass_bessell(pkg_data_name, name=None):
    """Bessell bandpasses have (1/energy) transmission units."""
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.AA, trans_unit=u.erg**-1,
                         normalize=True, name=name)


def load_bandpass_remote_aa(relpath, name=None):
    abspath = DATADIR.abspath(relpath)
    return read_bandpass(abspath, wave_unit=u.AA,
                         trim_level=BANDPASS_TRIM_LEVEL, name=name)


def load_bandpass_remote_nm(relpath, name=None):
    abspath = DATADIR.abspath(relpath)
    return read_bandpass(abspath, wave_unit=u.nm,
                         trim_level=BANDPASS_TRIM_LEVEL, name=name)


def load_bandpass_remote_um(relpath, name=None):
    abspath = DATADIR.abspath(relpath)
    return read_bandpass(abspath, wave_unit=u.micron,
                         trim_level=BANDPASS_TRIM_LEVEL, name=name)


def load_bandpass_remote_wfc3(relpath, name=None):
    abspath = DATADIR.abspath(relpath)
    _, wave, trans = np.loadtxt(abspath, unpack=True)
    return Bandpass(wave, trans, wave_unit=u.AA,
                    trim_level=BANDPASS_TRIM_LEVEL, name=name)


def tophat_bandpass_um(ctr, width, name=None):
    """Create a tophat Bandpass centered at `ctr` with width `width` (both
    in microns)."""

    wave = np.array([ctr - width / 2.0, ctr + width / 2.0])
    trans = np.array([1.0, 1.0])
    return Bandpass(wave, trans, wave_unit=u.micron, name=name)


# Bessell bandpasses (transmission is in units of (photons / erg))
bessell_meta = {
    'filterset': 'bessell',
    'reference': ('B90', '`Bessell 1990 <http://adsabs.harvard.edu/'
                  'abs/1990PASP..102.1181B>`__, Table 2'),
    'description': 'Representation of Johnson-Cousins UBVRI system'}

for name, fname in [('bessellux', 'bessell/bessell_ux.dat'),
                    ('bessellb', 'bessell/bessell_b.dat'),
                    ('bessellv', 'bessell/bessell_v.dat'),
                    ('bessellr', 'bessell/bessell_r.dat'),
                    ('besselli', 'bessell/bessell_i.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_bessell,
                                args=('data/bandpasses/' + fname,),
                                meta=bessell_meta)

# Shifted bessell filters used in SNLS3 (in units of photon / photon)
snls3_landolt_meta = {
    'filterset': 'snls3-landolt',
    'dataurl': 'http://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html',
    'retrieved': '13 February 2017',
    'description': 'Bessell bandpasses shifted as in JLA analysis',
    'reference': ('B14a',
                  '`Betoule et al. (2014) <http://adsabs.harvard.edu'
                  '/abs/2014A%26A...568A..22B>`__, Footnote 21')}
for name, fname in [
        ('standard::u', 'bandpasses/snls3-landolt/sux-shifted.dat'),
        ('standard::b', 'bandpasses/snls3-landolt/sb-shifted.dat'),
        ('standard::v', 'bandpasses/snls3-landolt/sv-shifted.dat'),
        ('standard::r', 'bandpasses/snls3-landolt/sr-shifted.dat'),
        ('standard::i', 'bandpasses/snls3-landolt/si-shifted.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(fname,), meta=snls3_landolt_meta)

des_meta = {
    'filterset': 'des',
    'retrieved': '22 March 2013',
    'description': 'Dark Energy Camera grizy filter set at airmass 1.3'}
for name, fname in [('desg', 'bandpasses/des/des_g.dat'),
                    ('desr', 'bandpasses/des/des_r.dat'),
                    ('desi', 'bandpasses/des/des_i.dat'),
                    ('desz', 'bandpasses/des/des_z.dat'),
                    ('desy', 'bandpasses/des/des_y.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(fname,), meta=des_meta)


sdss_meta = {
    'filterset': 'sdss',
    'reference': ('D10', '`Doi et al. 2010 <http://adsabs.harvard.edu/'
                  'abs/2010AJ....139.1628D>`__, Table 4'),
    'description': ('SDSS 2.5m imager at airmass 1.3 (including '
                    'atmosphere), normalized')}
for name, fname in [('sdssu', 'bandpasses/sdss/sdss_u.dat'),
                    ('sdssg', 'bandpasses/sdss/sdss_g.dat'),
                    ('sdssr', 'bandpasses/sdss/sdss_r.dat'),
                    ('sdssi', 'bandpasses/sdss/sdss_i.dat'),
                    ('sdssz', 'bandpasses/sdss/sdss_z.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(fname,),
                                meta=sdss_meta)

_BANDPASSES.alias('sdss::u', 'sdssu')
_BANDPASSES.alias('sdss::g', 'sdssg')
_BANDPASSES.alias('sdss::r', 'sdssr')
_BANDPASSES.alias('sdss::i', 'sdssi')
_BANDPASSES.alias('sdss::z', 'sdssz')


# HST ACS WFC bandpasses: remote
acs_meta = {'filterset': 'acs',
            'dataurl': 'http://www.stsci.edu/hst/acs/analysis/throughputs',
            'retrieved': 'direct download',
            'description': 'Hubble Space Telescope ACS WFC filters'}
for name, fname in [('f435w', 'bandpasses/acs-wfc/wfc_F435W.dat'),
                    ('f475w', 'bandpasses/acs-wfc/wfc_F475W.dat'),
                    ('f555w', 'bandpasses/acs-wfc/wfc_F555W.dat'),
                    ('f606w', 'bandpasses/acs-wfc/wfc_F606W.dat'),
                    ('f625w', 'bandpasses/acs-wfc/wfc_F625W.dat'),
                    ('f775w', 'bandpasses/acs-wfc/wfc_F775W.dat'),
                    # TODO: 814 filter from STScI has multiple identical
                    # wavelength values.
                    # ('f814w', 'bandpasses/acs-wfc/wfc_F814W.dat'),
                    ('f850lp', 'bandpasses/acs-wfc/wfc_F850LP.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(fname,), meta=acs_meta)

_BANDPASSES.alias('acswf::f606w', 'f606w')
_BANDPASSES.alias('acswf::f775w', 'f775w')
_BANDPASSES.alias('acswf::f850lp', 'f850lp')


# HST NICMOS NIC2 bandpasses: remote
nicmos_meta = {'filterset': 'nicmos-nic2',
               'dataurl': 'http://www.stsci.edu/hst/',
               'retrieved': '05 Aug 2014',
               'description': 'Hubble Space Telescope NICMOS2 filters'}
for name, fname in [
        ('nicf110w', 'bandpasses/nicmos-nic2/hst_nicmos_nic2_f110w.dat'),
        ('nicf160w', 'bandpasses/nicmos-nic2/hst_nicmos_nic2_f160w.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(fname,), meta=nicmos_meta)

_BANDPASSES.alias('nicmos2::f110w', 'nicf110w')
_BANDPASSES.alias('nicmos2::f160w', 'nicf160w')


# WFC3 IR bandpasses: remote
wfc3ir_meta = {'filterset': 'wfc3-ir',
               'dataurl': 'http://www.stsci.edu/hst/wfc3/ins_performance/'
                          'throughputs/Throughput_Tables',
               'retrieved': 'direct download',
               'description': 'Hubble Space Telescope WFC3 IR filters'}
for name, fname in [('f098m', 'bandpasses/wfc3-ir/f098m.IR.tab'),
                    ('f105w', 'bandpasses/wfc3-ir/f105w.IR.tab'),
                    ('f110w', 'bandpasses/wfc3-ir/f110w.IR.tab'),
                    ('f125w', 'bandpasses/wfc3-ir/f125w.IR.tab'),
                    ('f127m', 'bandpasses/wfc3-ir/f127m.IR.tab'),
                    ('f139m', 'bandpasses/wfc3-ir/f139m.IR.tab'),
                    ('f140w', 'bandpasses/wfc3-ir/f140w.IR.tab'),
                    ('f153m', 'bandpasses/wfc3-ir/f153m.IR.tab'),
                    ('f160w', 'bandpasses/wfc3-ir/f160w.IR.tab')]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_wfc3,
                                args=(fname,), meta=wfc3ir_meta)


wfc3uvis_meta = {'filterset': 'wfc3-uvis',
                 'dataurl': 'http://www.stsci.edu/hst/wfc3/ins_performance/'
                            'throughputs/Throughput_Tables',
                 'retrieved': 'direct download',
                 'description': ('Hubble Space Telescope WFC3 UVIS filters '
                                 '(CCD 2)')}
for name, fname in [('f218w', "bandpasses/wfc3-uvis/f218w.UVIS2.tab"),
                    ('f225w', "bandpasses/wfc3-uvis/f225w.UVIS2.tab"),
                    ('f275w', "bandpasses/wfc3-uvis/f275w.UVIS2.tab"),
                    ('f300x', "bandpasses/wfc3-uvis/f300x.UVIS2.tab"),
                    ('f336w', "bandpasses/wfc3-uvis/f336w.UVIS2.tab"),
                    ('f350lp', "bandpasses/wfc3-uvis/f350lp.UVIS2.tab"),
                    ('f390w', "bandpasses/wfc3-uvis/f390w.UVIS2.tab"),
                    ('f689m', "bandpasses/wfc3-uvis/f689m.UVIS2.tab"),
                    ('f763m', "bandpasses/wfc3-uvis/f763m.UVIS2.tab"),
                    ('f845m', "bandpasses/wfc3-uvis/f845m.UVIS2.tab"),
                    ('f438w', "bandpasses/wfc3-uvis/f438w.UVIS2.tab"),
                    ('uvf475w', "bandpasses/wfc3-uvis/f475w.UVIS2.tab"),
                    ('uvf555w', "bandpasses/wfc3-uvis/f555w.UVIS2.tab"),
                    ('uvf606w', "bandpasses/wfc3-uvis/f606w.UVIS2.tab"),
                    ('uvf625w', "bandpasses/wfc3-uvis/f625w.UVIS2.tab"),
                    ('uvf775w', "bandpasses/wfc3-uvis/f775w.UVIS2.tab"),
                    ('uvf814w', "bandpasses/wfc3-uvis/f814w.UVIS2.tab"),
                    ('uvf850lp', "bandpasses/wfc3-uvis/f850lp.UVIS2.tab")]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_wfc3,
                                args=(fname,), meta=wfc3uvis_meta)


# Kepler
kepler_meta = {
    'filterset': 'kepler',
    'retrieved': 'direct download',
    'description': 'Bandpass for the Kepler spacecraft',
    'dataurl': 'http://keplergo.arc.nasa.gov/CalibrationResponse.shtml'}
_BANDPASSES.register_loader(
    'kepler', load_bandpass_remote_nm,
    args=("bandpasses/kepler/kepler_response_hires1.txt",),
    meta=kepler_meta)


csp_meta = {
    'filterset': 'csp',
    'retrieved': '8 Feb 2017',
    'description': ('Carnegie Supernova Project filters (Swope+DuPont '
                    'Telescopes) updated 6 Oct 2016'),
    'dataurl': 'http://csp.obs.carnegiescience.edu/data/filters'}
for name, fname in [
        ('cspb',     'bandpasses/csp/B_tel_ccd_atm_ext_1.2.dat'),
        ('csphs',    'bandpasses/csp/H_SWO_TAM_scan_atm.dat'),
        ('csphd',    'bandpasses/csp/H_DUP_TAM_scan_atm.dat'),
        ('cspjs',    'bandpasses/csp/J_SWO_TAM_scan_atm.dat'),
        ('cspjd',    'bandpasses/csp/J_DUP_TAM_scan_atm.dat'),
        ('cspv3009', 'bandpasses/csp/V_LC3009_tel_ccd_atm_ext_1.2.dat'),
        ('cspv3014', 'bandpasses/csp/V_LC3014_tel_ccd_atm_ext_1.2.dat'),
        ('cspv9844', 'bandpasses/csp/V_LC9844_tel_ccd_atm_ext_1.2.dat'),
        ('cspys',    'bandpasses/csp/Y_SWO_TAM_scan_atm.dat'),
        ('cspyd',    'bandpasses/csp/Y_DUP_TAM_scan_atm.dat'),
        ('cspg',     'bandpasses/csp/g_tel_ccd_atm_ext_1.2.dat'),
        ('cspi',     'bandpasses/csp/i_tel_ccd_atm_ext_1.2.dat'),
        ('cspk',     'bandpasses/csp/kfilter'),
        ('cspr',     'bandpasses/csp/r_tel_ccd_atm_ext_1.2.dat'),
        ('cspu',     'bandpasses/csp/u_tel_ccd_atm_ext_1.2.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(fname,), meta=csp_meta)

_BANDPASSES.alias('swope2::u', 'cspu')
_BANDPASSES.alias('swope2::b', 'cspb')
_BANDPASSES.alias('swope2::g', 'cspg')
_BANDPASSES.alias('swope2::v', 'cspv3014')
_BANDPASSES.alias('swope2::v1', 'cspv3009')
_BANDPASSES.alias('swope2::v2', 'cspv9844')
_BANDPASSES.alias('swope2::r', 'cspr')
_BANDPASSES.alias('swope2::i', 'cspi')
_BANDPASSES.alias('swope2::y', 'cspys')
_BANDPASSES.alias('swope2::j', 'cspjs')
_BANDPASSES.alias('swope2::h', 'csphs')


jwst_nircam_meta = {'filterset': 'jwst-nircam',
                    'dataurl': 'http://www.stsci.edu/jwst/instruments/nircam'
                               '/instrumentdesign/filters',
                    'retrieved': '09 Sep 2014',
                    'description': 'James Webb Space Telescope NIRCAM '
                                   'Wide+Medium filters'}
for name, fname in [('f070w', 'bandpasses/nircam/jwst_nircam_f070w.dat'),
                    ('f090w', 'bandpasses/nircam/jwst_nircam_f090w.dat'),
                    ('f115w', 'bandpasses/nircam/jwst_nircam_f115w.dat'),
                    ('f150w', 'bandpasses/nircam/jwst_nircam_f150w.dat'),
                    ('f200w', 'bandpasses/nircam/jwst_nircam_f200w.dat'),
                    ('f277w', 'bandpasses/nircam/jwst_nircam_f277w.dat'),
                    ('f356w', 'bandpasses/nircam/jwst_nircam_f356w.dat'),
                    ('f444w', 'bandpasses/nircam/jwst_nircam_f444w.dat'),
                    ('f140m', 'bandpasses/nircam/jwst_nircam_f140m.dat'),
                    ('f162m', 'bandpasses/nircam/jwst_nircam_f162m.dat'),
                    ('f182m', 'bandpasses/nircam/jwst_nircam_f182m.dat'),
                    ('f210m', 'bandpasses/nircam/jwst_nircam_f210m.dat'),
                    ('f250m', 'bandpasses/nircam/jwst_nircam_f250m.dat'),
                    ('f300m', 'bandpasses/nircam/jwst_nircam_f300m.dat'),
                    ('f335m', 'bandpasses/nircam/jwst_nircam_f335m.dat'),
                    ('f360m', 'bandpasses/nircam/jwst_nircam_f360m.dat'),
                    ('f410m', 'bandpasses/nircam/jwst_nircam_f410m.dat'),
                    ('f430m', 'bandpasses/nircam/jwst_nircam_f430m.dat'),
                    ('f460m', 'bandpasses/nircam/jwst_nircam_f460m.dat'),
                    ('f480m', 'bandpasses/nircam/jwst_nircam_f480m.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_um,
                                args=(fname,), meta=jwst_nircam_meta)


jwst_miri_meta = {
    'filterset': 'jwst-miri',
    'dataurl': ('http://ircamera.as.arizona.edu/MIRI/'
                'ImPCE_TN-00072-ATC-Iss2.xlsx'),
    'retrieved': '16 Feb 2017',
    'description': 'James Webb Space Telescope MIRI filters'}
for name in ['f560w', 'f770w', 'f1000w', 'f1130w', 'f1280w',
             'f1500w', 'f1800w', 'f2100w', 'f2550w']:
    fname = "bandpasses/miri/jwst_miri_{}.dat".format(name)
    _BANDPASSES.register_loader(name, load_bandpass_remote_um,
                                args=(fname,), meta=jwst_miri_meta)


jwst_miri_meta2 = {'filterset': 'jwst-miri-tophat',
                   'dataurl': ('http://www.stsci.edu/jwst/instruments/miri/'
                               'instrumentdesign/filters'),
                   'retrieved': '09 Sep 2014',
                   'description': ('James Webb Space Telescope MIRI '
                                   'filters (idealized tophat)')}
for name, ctr, width in [('f1065c', 10.65, 0.53),
                         ('f1140c', 11.4, 0.57),
                         ('f1550c', 15.5, 0.78),
                         ('f2300c', 23., 4.6)]:
    _BANDPASSES.register_loader(name, tophat_bandpass_um,
                                args=(ctr, width), meta=jwst_miri_meta2)


# LSST bandpasses
lsst_meta = {'filterset': 'lsst',
             'dataurl': ('https://github.com/lsst/throughputs/tree/'
                         '7632edaa9e93d06576e34a065ea4622de8cc48d0/baseline'),
             'retrieved': '16 Nov 2016',
             'description': 'LSST baseline total throughputs, v1.1.'}
for letter in ['u', 'g', 'r', 'i', 'z', 'y']:
    name = 'lsst' + letter
    relpath = 'bandpasses/lsst/total_{}.dat'.format(letter)
    _BANDPASSES.register_loader(name, load_bandpass_remote_nm,
                                args=(relpath,), meta=lsst_meta)

# Keplercam
keplercam_meta = {
    'filterset': 'keplercam',
    'dataurl': 'http://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html',
    'retrieved': '13 Feb 2017',
    'description': 'Keplercam transmissions as used in JLA'}
for name, fname in [('keplercam::us', 'bandpasses/keplercam/Us_Keplercam.txt'),
                    ('keplercam::b', 'bandpasses/keplercam/B_Keplercam.txt'),
                    ('keplercam::v', 'bandpasses/keplercam/V_Keplercam.txt'),
                    ('keplercam::r', 'bandpasses/keplercam/r_Keplercam.txt'),
                    ('keplercam::i', 'bandpasses/keplercam/i_Keplercam.txt')]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(fname,), meta=keplercam_meta)

# 4shooter
fourshooter_meta = {
    'filterset': '4shooter2',
    'dataurl': 'http://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html',
    'retrieved': '13 Feb 2017',
    'description': '4Shooter filters as used in JLA'}
for name, fname in [('4shooter2::us', 'bandpasses/4shooter2/Us_4Shooter2.txt'),
                    ('4shooter2::b', 'bandpasses/4shooter2/B_4Shooter2.txt'),
                    ('4shooter2::v', 'bandpasses/4shooter2/V_4Shooter2.txt'),
                    ('4shooter2::r', 'bandpasses/4shooter2/R_4Shooter2.txt'),
                    ('4shooter2::i', 'bandpasses/4shooter2/I_4Shooter2.txt')]:

    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(fname,), meta=fourshooter_meta)

roman_meta = {
     'filterset': 'roman-wfi',
     'dataurl': 'https://roman.gsfc.nasa.gov/science/WFI_technical.html',
     'retrieved': '30 Nov 2020',
     'description': 'Roman filters from Jeff Kruk: '
     'Roman_effarea_20201130.txt '}
for name, fname in [('f062', 'roman_f062.dat'),   # R
                    ('f087', 'roman_f087.dat'),  # Z
                    ('f106', 'roman_f106.dat'),  # Y
                    ('f129', 'roman_f129.dat'),  # J
                    ('f158', 'roman_f158.dat'),  # H
                    ('f184', 'roman_f184.dat'),  # F
                    ('f213', 'roman_f213.dat'),  # K
                    ('f146', 'roman_f146.dat')]:  # Wide
    _BANDPASSES.register_loader(name, load_bandpass_remote_um,
                                args=('bandpasses/roman-wfi/' + fname,),
                                meta=roman_meta)

# ZTF
ztf_meta = {
    'filterset': 'ztf',
    'retrieved': '7 Jun 2018',
    'description': 'ZTF filters from Uli Feindt. No atmospheric correction.'}
for name, fname in [('ztfg', 'bandpasses/ztf/P48_g.dat'),
                    ('ztfr', 'bandpasses/ztf/P48_R.dat'),
                    ('ztfi', 'bandpasses/ztf/P48_I.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(fname,),
                                meta=ztf_meta)


# Swift UVOT
swift_meta = {
    'filterset': 'swift-uvot',
    'retrieved': '19 May 2020',
    'description': "Swift UVOT filters retreieved from the Spanish Virtual "
                   "Observatory filter profile service."
}

for name, fname in [('uvot::b', 'bandpasses/swift/Swift_UVOT.B.dat'),
                    ('uvot::u', 'bandpasses/swift/Swift_UVOT.U.dat'),
                    ('uvot::uvm2', 'bandpasses/swift/Swift_UVOT.UVM2.dat'),
                    ('uvot::uvw1', 'bandpasses/swift/Swift_UVOT.UVW1.dat'),
                    ('uvot::uvw2', 'bandpasses/swift/Swift_UVOT.UVW2.dat'),
                    ('uvot::v', 'bandpasses/swift/Swift_UVOT.V.dat'),
                    ('uvot::white', 'bandpasses/swift/Swift_UVOT.white.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(fname,),
                                meta=swift_meta)

# Pan-STARRS1
ps1_meta = {
    'filterset': 'ps1',
    'dataurl': 'https://ipp.ifa.hawaii.edu/ps1.filters/',
    'reference': ('T12', '`Tonry 2012 <http://adsabs.harvard.edu/'
                  'abs/2012ApJ...750...99T>`__, Table 3'),
    'retrieved': '17 Aug 2021',
    'description': 'Pan-STARRS1 filters at airmass 1.2'
}
for filter_id in ['open', 'g', 'r', 'i', 'z', 'y', 'w']:
    name = 'ps1::{}'.format(filter_id)
    fname = "bandpasses/ps1/ps1_{}.dat".format(filter_id)
    _BANDPASSES.register_loader(name, load_bandpass_remote_nm,
                                args=(fname,),
                                meta=ps1_meta)

# ATLAS
atlas_meta = {
    'filterset': 'atlas',
    'retrieved': '28 Dec 2021',
    'dataurl': ('http://svo2.cab.inta-csic.es/svo/theory/fps/getdata.php?'
                'format=ascii&id=Misc'),
    'description': ('ATLAS filters from SVO (includes filter, optics,'
                    'detector and atmosphere.)')}
for filt in ['Cyan', 'Orange']:
    name = 'atlas' + filt[0].lower()
    relpath = 'bandpasses/atlas/Atlas.{}'.format(filt)
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(relpath,), meta=atlas_meta)

# 2MASS
twomass_meta = {
    'filterset': '2MASS',
    'retrieved': '1 Feb 2022',
    'dataurl': ('http://svo2.cab.inta-csic.es/svo/theory/fps/getdata.php?'
                'format=ascii&id=Misc'),
    'description': ('2MASS filters from SVO (includes filter, instrument,'
                    'and atmosphere.)')}
for filt in ['J', 'H', 'Ks']:
    name = '2mass' + filt.lower()
    relpath = 'bandpasses/2mass/2mass.{}'.format(filt)
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(relpath,), meta=twomass_meta)

# Gaia
gaia_meta = {
    'filterset': 'gaia',
    'retrieved': '27 Sep 2022',
    'dataurl': ('http://svo2.cab.inta-csic.es/svo/theory/fps/getdata.php?'
                'format=ascii&id=Misc'),
    'description': ('eDR3 Gaia filters from SVO (includes filter, instrument,'
                    'and optics.)')}
for filt in ['Gbp', 'G', 'Grp', 'Grvs']:
    name = 'gaia::' + filt.lower()
    relpath = 'bandpasses/gaia/gaia.{}'.format(filt)
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(relpath,), meta=gaia_meta)

# TESS
tess_meta = {                                                         
    'filterset': 'tess',       
    'retrieved': '7 March 2023',
    'dataurl': ('http://svo2.cab.inta-csic.es/svo/theory/fps/getdata.php?'
                'format=ascii&id=Misc'),
    'description': 'TESS filter from SVO (includes filter and instrument)'
}                
for filt in ['Red']:
    name = 'tess'
    relpath = 'bandpasses/tess/tess.{}'.format(filt)
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(relpath,), meta=tess_meta)


# =============================================================================
# interpolators

megacam_meta = {'filterset': 'megacampsf'}


def load_megacampsf(letter, name=None):
    abspath = DATADIR.abspath('bandpasses/megacampsf', isdir=True)
    return snfitio.read_snfit_bandpass_interpolator(abspath, letter, name=name)


for letter in ('u', 'g', 'r', 'i', 'z', 'y'):
    _BANDPASS_INTERPOLATORS.register_loader('megacampsf::' + letter,
                                            load_megacampsf, args=(letter,),
                                            meta=megacam_meta)

# =============================================================================
# Sources


def load_timeseries_ascii(relpath, zero_before=False, time_spline_degree=3,
                          name=None, version=None):
    abspath = DATADIR.abspath(relpath)
    phase, wave, flux = io.read_griddata_ascii(abspath)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version,
                            zero_before=zero_before,
                            time_spline_degree=time_spline_degree)


def load_timeseries_fits(relpath, name=None, version=None):
    abspath = DATADIR.abspath(relpath)
    phase, wave, flux = io.read_griddata_fits(abspath)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version)


def load_timeseries_fits_local(pkg_data_name, name=None, version=None):
    fname = get_pkg_data_filename(pkg_data_name)
    phase, wave, flux = io.read_griddata_fits(fname)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version)


def load_salt2model(relpath, name=None, version=None):
    abspath = DATADIR.abspath(relpath, isdir=True)
    return SALT2Source(modeldir=abspath, name=name, version=version)


def load_salt3model(relpath, name=None, version=None):
    abspath = DATADIR.abspath(relpath, isdir=True)
    return SALT3Source(modeldir=abspath, name=name, version=version)


def load_2011fe(relpath, name=None, version=None):

    # filter warnings about RADESYS keyword in files
    warnings.filterwarnings('ignore', category=wcs.FITSFixedWarning,
                            append=True)

    abspath = DATADIR.abspath(relpath, isdir=True)

    phasestrs = []
    spectra = []
    disp = None
    for fname in os.listdir(abspath):
        if fname[-4:] == '.fit':
            hdulist = fits.open(join(abspath, fname))
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

    return TimeSeriesSource(phases, disp, flux, time_spline_degree=1,
                            name=name, version=version)


# Nugent models
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

nugent_models = [('sn1a', '1.2', 'SN Ia', n02ref, 3),
                 ('sn91t', '1.1', 'SN Ia', s04ref, 3),
                 ('sn91bg', '1.1', 'SN Ia', n02ref, 3),
                 ('sn1bc', '1.1', 'SN Ib/c', l05ref, 3),
                 ('hyper', '1.2', 'SN Ib/c', l05ref, 1),
                 ('sn2p', '1.2', 'SN IIP', g99ref, 1),
                 ('sn2l', '1.2', 'SN IIL', g99ref, 1),
                 ('sn2n', '2.1', 'SN IIn', g99ref, 1)]

for suffix, ver, sntype, ref, time_spline_degree in nugent_models:
    name = "nugent-" + suffix
    relpath = "models/nugent/{0}_flux.v{1}.dat".format(suffix, ver)
    _SOURCES.register_loader(name, load_timeseries_ascii,
                             args=(relpath, False, time_spline_degree),
                             version=ver,
                             meta={'url': website, 'type': sntype,
                                   'subclass': subclass, 'reference': ref})

# Sako et al 2011 models
ref = ('S11', 'Sako et al. 2011 '
       '<http://adsabs.harvard.edu/abs/2011ApJ...738..162S>')
website = 'http://sdssdp62.fnal.gov/sdsssn/SNANA-PUBLIC/'
subclass = '`~sncosmo.TimeSeriesSource`'
note = "extracted from SNANA's SNDATA_ROOT on 29 March 2013."

for name, sntype, fn in [('s11-2004hx', 'SN IIL/P', 'S11_SDSS-000018.SED'),
                         ('s11-2005lc', 'SN IIP', 'S11_SDSS-001472.SED'),
                         ('s11-2005hl', 'SN Ib', 'S11_SDSS-002000.SED'),
                         ('s11-2005hm', 'SN Ib', 'S11_SDSS-002744.SED'),
                         ('s11-2005gi', 'SN IIP', 'S11_SDSS-003818.SED'),
                         ('s11-2006fo', 'SN Ic', 'S11_SDSS-013195.SED'),
                         ('s11-2006jo', 'SN Ib', 'S11_SDSS-014492.SED'),
                         ('s11-2006jl', 'SN IIP', 'S11_SDSS-014599.SED')]:
    meta = {'url': website, 'type': sntype, 'subclass': subclass,
            'reference': ref, 'note': note}
    _SOURCES.register_loader(name, load_timeseries_ascii,
                             args=('models/sako/' + fn,), version='1.0',
                             meta=meta)

# Hsiao models
meta = {'url': 'http://csp.obs.carnegiescience.edu/data/snpy',
        'type': 'SN Ia',
        'subclass': '`~sncosmo.TimeSeriesSource`',
        'reference': ('H07', 'Hsiao et al. 2007 '
                      '<http://adsabs.harvard.edu/abs/2007ApJ...663.1187H>'),
        'note': 'extracted from the SNooPy package on 21 Dec 2012.'}
for version, fn in [('1.0', 'Hsiao_SED.fits'),
                    ('2.0', 'Hsiao_SED_V2.fits'),
                    ('3.0', 'Hsiao_SED_V3.fits')]:
    _SOURCES.register_loader('hsiao', load_timeseries_fits,
                             args=('models/hsiao/' + fn,), version=version,
                             meta=meta)

# subsampled version of Hsiao v3.0, for testing purposes.
_SOURCES.register_loader('hsiao-subsampled',
                         load_timeseries_fits_local,
                         args=('data/models/Hsiao_SED_V3_subsampled.fits',),
                         version='3.0', meta=meta)

# SALT2 models
website = 'http://supernovae.in2p3.fr/salt/doku.php?id=salt_templates'
g10ref = ('G10', 'Guy et al. 2010 '
          '<http://adsabs.harvard.edu/abs/2010A%26A...523A...7G>')
b14ref = ('B14b', 'Betoule et al. 2014 '
          '<http://adsabs.harvard.edu/abs/2014A%26A...568A..22B>')
for topdir, ver, ref in [('salt2-2-0', '2.0', g10ref),
                         ('salt2-4', '2.4', b14ref)]:
    meta = {'type': 'SN Ia', 'subclass': '`~sncosmo.SALT2Source`',
            'url': website, 'reference': ref}
    _SOURCES.register_loader('salt2', load_salt2model,
                             args=('models/salt2/'+topdir,),
                             version=ver, meta=meta)

# SALT2 extended
meta = {'type': 'SN Ia',
        'subclass': '`~sncosmo.SALT2Source`',
        'url': 'http://sdssdp62.fnal.gov/sdsssn/SNANA-PUBLIC/',
        'note': "extracted from SNANA's SNDATA_ROOT on 15 August 2013."}
_SOURCES.register_loader('salt2-extended', load_salt2model,
                         args=('models/snana/salt2_extended',), version='1.0',
                         meta=meta)

ref = ('SNSEDExtend', 'Pierel et al. 2018'
       '<https://arxiv.org/abs/1808.02534>')
meta = {'type': 'SN Ia',
        'subclass': '`~sncosmo.SALT2Source`', 'ref': ref}
_SOURCES.register_loader('salt2-extended', load_salt2model,
                         args=('models/pierel/salt2-extended',), version='2.0',
                         meta=meta)

# SALT3-NIR
meta = {'type': 'SN Ia',
        'subclass': '`~sncosmo.SALT3Source`',
        'url': 'https://doi.org/10.5281/zenodo.7068818',
        'note': """See Pierel et al. 2022, ApJ."""}
_SOURCES.register_loader('salt3-nir', load_salt3model,
                         args=('models/salt3-nir/salt3nir-p22',),
                         version='1.0',
                         meta=meta)

# SALT3
meta = {'type': 'SN Ia',
        'subclass': '`~sncosmo.SALT3Source`',
        'url': 'https://salt3.readthedocs.io/en/latest/',
        'note': """See Kenworthy et al. 2021, ApJ, 923, 265K.
Revised with Fragilistic calibration (Brout et al., 2022)"""}
_SOURCES.register_loader('salt3', load_salt3model,
                         args=('models/salt3/salt3-f22',), version='2.0',
                         meta=meta)

meta = {'type': 'SN Ia',
        'subclass': '`~sncosmo.SALT2Source`',
        'url': 'http://snana.uchicago.edu/',
        'note': "extracted from SNANA's SNDATA_ROOT on 24 April 2018. SALT2"
        " model with wide wavelength range, Hounsell et al. 2017",
        'reference': ('H17', 'Hounsell et al. 2017 '
                      '<http://adsabs.harvard.edu/abs/2017arXiv170201747H>')}
_SOURCES.register_loader('salt2-extended-h17', load_salt2model,
                         args=('models/snana/salt2-h17',),
                         version='1.0', meta=meta)

# Alias to 'salt2-h17' for backwards-compatibility
_SOURCES.alias('salt2-h17', 'salt2-extended-h17', new_version='1.0',
               existing_version='1.0')


# 2011fe
meta = {'type': 'SN Ia',
        'subclass': '`~sncosmo.TimeSeriesSource`',
        'url': 'http://snfactory.lbl.gov/snf/data',
        'reference': ('P13', 'Pereira et al. 2013 '
                      '<http://adsabs.harvard.edu/abs/2013A%26A...554A..27P>')}
_SOURCES.register_loader('snf-2011fe', load_2011fe, version='1.0',
                         args=('models/snf/SN2011fe',), meta=meta)


# SNANA CC SN models
url = 'http://das.sdss2.org/ge/sample/sdsssn/SNANA-PUBLIC/'
subclass = '`~sncosmo.TimeSeriesSource`'
ref = ('SNANA', 'Kessler et al. 2009 '
       '<http://adsabs.harvard.edu/abs/2009PASP..121.1028K>')
note = "extracted from SNANA's SNDATA_ROOT on 5 August 2014."

# 'PSNID' denotes that model is used in PSNID.
models = [('snana-2004fe', 'SN Ic', 'CSP-2004fe.SED'),
          ('snana-2004gq', 'SN Ic', 'CSP-2004gq.SED'),
          ('snana-sdss004012', 'SN Ic', 'SDSS-004012.SED'),  # no IAU name
          ('snana-2006fo', 'SN Ic', 'SDSS-013195.SED'),  # PSNID
          ('snana-sdss014475', 'SN Ic', 'SDSS-014475.SED'),  # no IAU name
          ('snana-2006lc', 'SN Ic', 'SDSS-015475.SED'),
          ('snana-2007ms', 'SN II-pec', 'SDSS-017548.SED'),  # type Ic in SNANA
          ('snana-04d1la', 'SN Ic', 'SNLS-04D1la.SED'),
          ('snana-04d4jv', 'SN Ic', 'SNLS-04D4jv.SED'),
          ('snana-2004gv', 'SN Ib', 'CSP-2004gv.SED'),
          ('snana-2006ep', 'SN Ib', 'CSP-2006ep.SED'),
          ('snana-2007Y', 'SN Ib', 'CSP-2007Y.SED'),
          ('snana-2004ib', 'SN Ib', 'SDSS-000020.SED'),
          ('snana-2005hm', 'SN Ib', 'SDSS-002744.SED'),  # PSNID
          ('snana-2006jo', 'SN Ib', 'SDSS-014492.SED'),  # PSNID
          ('snana-2007nc', 'SN Ib', 'SDSS-019323.SED'),
          ('snana-2004hx', 'SN IIP', 'SDSS-000018.SED'),  # PSNID
          ('snana-2005gi', 'SN IIP', 'SDSS-003818.SED'),  # PSNID
          ('snana-2006gq', 'SN IIP', 'SDSS-013376.SED'),
          ('snana-2006kn', 'SN IIP', 'SDSS-014450.SED'),
          ('snana-2006jl', 'SN IIP', 'SDSS-014599.SED'),  # PSNID
          ('snana-2006iw', 'SN IIP', 'SDSS-015031.SED'),
          ('snana-2006kv', 'SN IIP', 'SDSS-015320.SED'),
          ('snana-2006ns', 'SN IIP', 'SDSS-015339.SED'),
          ('snana-2007iz', 'SN IIP', 'SDSS-017564.SED'),
          ('snana-2007nr', 'SN IIP', 'SDSS-017862.SED'),
          ('snana-2007kw', 'SN IIP', 'SDSS-018109.SED'),
          ('snana-2007ky', 'SN IIP', 'SDSS-018297.SED'),
          ('snana-2007lj', 'SN IIP', 'SDSS-018408.SED'),
          ('snana-2007lb', 'SN IIP', 'SDSS-018441.SED'),
          ('snana-2007ll', 'SN IIP', 'SDSS-018457.SED'),
          ('snana-2007nw', 'SN IIP', 'SDSS-018590.SED'),
          ('snana-2007ld', 'SN IIP', 'SDSS-018596.SED'),
          ('snana-2007md', 'SN IIP', 'SDSS-018700.SED'),
          ('snana-2007lz', 'SN IIP', 'SDSS-018713.SED'),
          ('snana-2007lx', 'SN IIP', 'SDSS-018734.SED'),
          ('snana-2007og', 'SN IIP', 'SDSS-018793.SED'),
          ('snana-2007ny', 'SN IIP', 'SDSS-018834.SED'),
          ('snana-2007nv', 'SN IIP', 'SDSS-018892.SED'),
          ('snana-2007pg', 'SN IIP', 'SDSS-020038.SED'),
          ('snana-2006ez', 'SN IIn', 'SDSS-012842.SED'),
          ('snana-2006ix', 'SN IIn', 'SDSS-013449.SED')]
for name, sntype, fn in models:
    relpath = 'models/snana/' + fn
    meta = {'url': url, 'subclass': subclass, 'type': sntype, 'ref': ref,
            'note': note}
    _SOURCES.register_loader(name, load_timeseries_ascii,
                             args=(relpath,), version='1.0', meta=meta)

# P18
p18Models_CC = [('snana-2004fe', 'SN Ic', 'CSP-2004fe.SED'),
                ('snana-2004gq', 'SN Ic', 'CSP-2004gq.SED'),
                ('snana-sdss004012', 'SN Ic', 'SDSS-004012.SED'),  # no IAU
                ('snana-2006fo', 'SN Ic', 'SDSS-013195.SED'),  # PSNID
                ('snana-sdss014475', 'SN Ic', 'SDSS-014475.SED'),  # no IAU
                ('snana-2006lc', 'SN Ic', 'SDSS-015475.SED'),
                ('snana-2007ms', 'SN II-pec', 'SDSS-017548.SED'),
                ('snana-04d1la', 'SN Ic', 'SNLS-04D1la.SED'),
                ('snana-04d4jv', 'SN Ic', 'SNLS-04D4jv.SED'),
                ('snana-2004gv', 'SN Ib', 'CSP-2004gv.SED'),
                ('snana-2006ep', 'SN Ib', 'CSP-2006ep.SED'),
                ('snana-2007Y', 'SN Ib', 'CSP-2007Y.SED'),
                ('snana-2004ib', 'SN Ib', 'SDSS-000020.SED'),
                ('snana-2005hm', 'SN Ib', 'SDSS-002744.SED'),  # PSNID
                ('snana-2006jo', 'SN Ib', 'SDSS-014492.SED'),  # PSNID
                ('snana-2007nc', 'SN Ib', 'SDSS-019323.SED'),
                ('snana-2004hx', 'SN IIP', 'SDSS-000018.SED'),  # PSNID
                ('snana-2005gi', 'SN IIP', 'SDSS-003818.SED'),  # PSNID
                ('snana-2006gq', 'SN IIP', 'SDSS-013376.SED'),
                ('snana-2006kn', 'SN IIP', 'SDSS-014450.SED'),
                ('snana-2006jl', 'SN IIP', 'SDSS-014599.SED'),  # PSNID
                ('snana-2006iw', 'SN IIP', 'SDSS-015031.SED'),
                ('snana-2006kv', 'SN IIP', 'SDSS-015320.SED'),
                ('snana-2006ns', 'SN IIP', 'SDSS-015339.SED'),
                ('snana-2007iz', 'SN IIP', 'SDSS-017564.SED'),
                ('snana-2007nr', 'SN IIP', 'SDSS-017862.SED'),
                ('snana-2007kw', 'SN IIP', 'SDSS-018109.SED'),
                ('snana-2007ky', 'SN IIP', 'SDSS-018297.SED'),
                ('snana-2007lj', 'SN IIP', 'SDSS-018408.SED'),
                ('snana-2007lb', 'SN IIP', 'SDSS-018441.SED'),
                ('snana-2007ll', 'SN IIP', 'SDSS-018457.SED'),
                ('snana-2007nw', 'SN IIP', 'SDSS-018590.SED'),
                ('snana-2007ld', 'SN IIP', 'SDSS-018596.SED'),
                ('snana-2007md', 'SN IIP', 'SDSS-018700.SED'),
                ('snana-2007lz', 'SN IIP', 'SDSS-018713.SED'),
                ('snana-2007lx', 'SN IIP', 'SDSS-018734.SED'),
                ('snana-2007og', 'SN IIP', 'SDSS-018793.SED'),
                ('snana-2007ny', 'SN IIP', 'SDSS-018834.SED'),
                ('snana-2007nv', 'SN IIP', 'SDSS-018892.SED'),
                ('snana-2007pg', 'SN IIP', 'SDSS-020038.SED')]

ref = ('SNSEDExtend', 'Pierel et al. 2018'
       '<https://arxiv.org/abs/1808.02534>')
for name, sntype, fn in p18Models_CC:
    relpath = 'models/pierel/' + fn
    meta = {'subclass': '`~sncosmo.TimeSeriesSource`', 'type': sntype,
            'ref': ref}
    _SOURCES.register_loader(name, load_timeseries_ascii,
                             args=(relpath,), version='2.0', meta=meta)

# V19
V19_CC_models = [
    ('v19-asassn14jb-corr', '1.0', 'SN II', 'V19_ASASSN14jb_HostExtCorr.SED'),
    ('v19-asassn14jb', '1.0', 'SN II', 'V19_ASASSN14jb_noHostExtCorr.SED'),
    ('v19-asassn15oz-corr', '1.0', 'SN II', 'V19_ASASSN15oz_HostExtCorr.SED'),
    ('v19-asassn15oz', '1.0', 'SN II', 'V19_ASASSN15oz_noHostExtCorr.SED'),
    ('v19-1987A-corr', '1.0', 'SN II', 'V19_SN1987A_HostExtCorr.SED'),
    ('v19-1987A', '1.0', 'SN II', 'V19_SN1987A_noHostExtCorr.SED'),
    ('v19-1993J-corr', '1.0', 'SN IIb', 'V19_SN1993J_HostExtCorr.SED'),
    ('v19-1993J', '1.0', 'SN IIb', 'V19_SN1993J_noHostExtCorr.SED'),
    ('v19-1994I-corr', '1.0', 'SN Ic', 'V19_SN1994I_HostExtCorr.SED'),
    ('v19-1994I', '1.0', 'SN Ic', 'V19_SN1994I_noHostExtCorr.SED'),
    ('v19-1998bw-corr', '1.0', 'SN Ic-BL', 'V19_SN1998bw_HostExtCorr.SED'),
    ('v19-1998bw', '1.0', 'SN Ic-BL', 'V19_SN1998bw_noHostExtCorr.SED'),
    ('v19-1999dn-corr', '1.0', 'SN IIb', 'V19_SN1999dn_HostExtCorr.SED'),
    ('v19-1999dn', '1.0', 'SN IIb', 'V19_SN1999dn_noHostExtCorr.SED'),
    ('v19-1999em-corr', '1.0', 'SN II', 'V19_SN1999em_HostExtCorr.SED'),
    ('v19-1999em', '1.0', 'SN II', 'V19_SN1999em_noHostExtCorr.SED'),
    ('v19-2002ap-corr', '1.0', 'SN Ic-BL', 'V19_SN2002ap_HostExtCorr.SED'),
    ('v19-2002ap', '1.0', 'SN Ic-BL', 'V19_SN2002ap_noHostExtCorr.SED'),
    ('v19-2004aw-corr', '1.0', 'SN Ic', 'V19_SN2004aw_HostExtCorr.SED'),
    ('v19-2004aw', '1.0', 'SN Ic', 'V19_SN2004aw_noHostExtCorr.SED'),
    ('v19-2004et-corr', '1.0', 'SN II', 'V19_SN2004et_HostExtCorr.SED'),
    ('v19-2004et', '1.0', 'SN II', 'V19_SN2004et_noHostExtCorr.SED'),
    ('v19-2004fe-corr', '1.0', 'SN Ic', 'V19_SN2004fe_HostExtCorr.SED'),
    ('v19-2004fe', '1.0', 'SN Ic', 'V19_SN2004fe_noHostExtCorr.SED'),
    ('v19-2004gq-corr', '1.0', 'SN Ib', 'V19_SN2004gq_HostExtCorr.SED'),
    ('v19-2004gq', '1.0', 'SN Ib', 'V19_SN2004gq_noHostExtCorr.SED'),
    ('v19-2004gt-corr', '1.0', 'SN Ic', 'V19_SN2004gt_HostExtCorr.SED'),
    ('v19-2004gt', '1.0', 'SN Ic', 'V19_SN2004gt_noHostExtCorr.SED'),
    ('v19-2004gv-corr', '1.0', 'SN Ib', 'V19_SN2004gv_HostExtCorr.SED'),
    ('v19-2004gv', '1.0', 'SN Ib', 'V19_SN2004gv_noHostExtCorr.SED'),
    ('v19-2005bf-corr', '1.0', 'SN Ib', 'V19_SN2005bf_HostExtCorr.SED'),
    ('v19-2005bf', '1.0', 'SN Ib', 'V19_SN2005bf_noHostExtCorr.SED'),
    ('v19-2005hg-corr', '1.0', 'SN Ib', 'V19_SN2005hg_HostExtCorr.SED'),
    ('v19-2005hg', '1.0', 'SN Ib', 'V19_SN2005hg_noHostExtCorr.SED'),
    ('v19-2006T-corr', '1.0', 'SN IIb', 'V19_SN2006T_HostExtCorr.SED'),
    ('v19-2006T', '1.0', 'SN IIb', 'V19_SN2006T_noHostExtCorr.SED'),
    ('v19-2006aa-corr', '1.0', 'SN IIn', 'V19_SN2006aa_HostExtCorr.SED'),
    ('v19-2006aa', '1.0', 'SN IIn', 'V19_SN2006aa_noHostExtCorr.SED'),
    ('v19-2006aj-corr', '1.0', 'SN Ic-BL', 'V19_SN2006aj_HostExtCorr.SED'),
    ('v19-2006aj', '1.0', 'SN Ic-BL', 'V19_SN2006aj_noHostExtCorr.SED'),
    ('v19-2006ep-corr', '1.0', 'SN Ib', 'V19_SN2006ep_HostExtCorr.SED'),
    ('v19-2006ep', '1.0', 'SN Ib', 'V19_SN2006ep_noHostExtCorr.SED'),
    ('v19-2007Y-corr', '1.0', 'SN Ib', 'V19_SN2007Y_HostExtCorr.SED'),
    ('v19-2007Y', '1.0', 'SN Ib', 'V19_SN2007Y_noHostExtCorr.SED'),
    ('v19-2007gr-corr', '1.0', 'SN Ic', 'V19_SN2007gr_HostExtCorr.SED'),
    ('v19-2007gr', '1.0', 'SN Ic', 'V19_SN2007gr_noHostExtCorr.SED'),
    ('v19-2007od-corr', '1.0', 'SN II', 'V19_SN2007od_HostExtCorr.SED'),
    ('v19-2007od', '1.0', 'SN II', 'V19_SN2007od_noHostExtCorr.SED'),
    ('v19-2007pk-corr', '1.0', 'SN IIn', 'V19_SN2007pk_HostExtCorr.SED'),
    ('v19-2007pk', '1.0', 'SN IIn', 'V19_SN2007pk_noHostExtCorr.SED'),
    ('v19-2007ru-corr', '1.0', 'SN Ic-BL', 'V19_SN2007ru_HostExtCorr.SED'),
    ('v19-2007ru', '1.0', 'SN Ic-BL', 'V19_SN2007ru_noHostExtCorr.SED'),
    ('v19-2007uy-corr', '1.0', 'SN Ib', 'V19_SN2007uy_HostExtCorr.SED'),
    ('v19-2007uy', '1.0', 'SN Ib', 'V19_SN2007uy_noHostExtCorr.SED'),
    ('v19-2008D-corr', '1.0', 'SN Ib', 'V19_SN2008D_HostExtCorr.SED'),
    ('v19-2008D', '1.0', 'SN Ib', 'V19_SN2008D_noHostExtCorr.SED'),
    ('v19-2008aq-corr', '1.0', 'SN IIb', 'V19_SN2008aq_HostExtCorr.SED'),
    ('v19-2008aq', '1.0', 'SN IIb', 'V19_SN2008aq_noHostExtCorr.SED'),
    ('v19-2008ax-corr', '1.0', 'SN IIb', 'V19_SN2008ax_HostExtCorr.SED'),
    ('v19-2008ax', '1.0', 'SN IIb', 'V19_SN2008ax_noHostExtCorr.SED'),
    ('v19-2008bj-corr', '1.0', 'SN II', 'V19_SN2008bj_HostExtCorr.SED'),
    ('v19-2008bj', '1.0', 'SN II', 'V19_SN2008bj_noHostExtCorr.SED'),
    ('v19-2008bo-corr', '1.0', 'SN IIb', 'V19_SN2008bo_HostExtCorr.SED'),
    ('v19-2008bo', '1.0', 'SN IIb', 'V19_SN2008bo_noHostExtCorr.SED'),
    ('v19-2008fq-corr', '1.0', 'SN IIn', 'V19_SN2008fq_HostExtCorr.SED'),
    ('v19-2008fq', '1.0', 'SN IIn', 'V19_SN2008fq_noHostExtCorr.SED'),
    ('v19-2008in-corr', '1.0', 'SN II', 'V19_SN2008in_HostExtCorr.SED'),
    ('v19-2008in', '1.0', 'SN II', 'V19_SN2008in_noHostExtCorr.SED'),
    ('v19-2009N-corr', '1.0', 'SN II', 'V19_SN2009N_HostExtCorr.SED'),
    ('v19-2009N', '1.0', 'SN II', 'V19_SN2009N_noHostExtCorr.SED'),
    ('v19-2009bb-corr', '1.0', 'SN Ic-BL', 'V19_SN2009bb_HostExtCorr.SED'),
    ('v19-2009bb', '1.0', 'SN Ic-BL', 'V19_SN2009bb_noHostExtCorr.SED'),
    ('v19-2009bw-corr', '1.0', 'SN II', 'V19_SN2009bw_HostExtCorr.SED'),
    ('v19-2009bw', '1.0', 'SN II', 'V19_SN2009bw_noHostExtCorr.SED'),
    ('v19-2009dd-corr', '1.0', 'SN II', 'V19_SN2009dd_HostExtCorr.SED'),
    ('v19-2009dd', '1.0', 'SN II', 'V19_SN2009dd_noHostExtCorr.SED'),
    ('v19-2009ib-corr', '1.0', 'SN II', 'V19_SN2009ib_HostExtCorr.SED'),
    ('v19-2009ib', '1.0', 'SN II', 'V19_SN2009ib_noHostExtCorr.SED'),
    ('v19-2009ip-corr', '1.0', 'SN IIn', 'V19_SN2009ip_HostExtCorr.SED'),
    ('v19-2009ip', '1.0', 'SN IIn', 'V19_SN2009ip_noHostExtCorr.SED'),
    ('v19-2009iz-corr', '1.0', 'SN Ib', 'V19_SN2009iz_HostExtCorr.SED'),
    ('v19-2009iz', '1.0', 'SN Ib', 'V19_SN2009iz_noHostExtCorr.SED'),
    ('v19-2009jf-corr', '1.0', 'SN Ib', 'V19_SN2009jf_HostExtCorr.SED'),
    ('v19-2009jf', '1.0', 'SN Ib', 'V19_SN2009jf_noHostExtCorr.SED'),
    ('v19-2009kr-corr', '1.0', 'SN II', 'V19_SN2009kr_HostExtCorr.SED'),
    ('v19-2009kr', '1.0', 'SN II', 'V19_SN2009kr_noHostExtCorr.SED'),
    ('v19-2010al-corr', '1.0', 'SN IIn', 'V19_SN2010al_HostExtCorr.SED'),
    ('v19-2010al', '1.0', 'SN IIn', 'V19_SN2010al_noHostExtCorr.SED'),
    ('v19-2011bm-corr', '1.0', 'SN Ic', 'V19_SN2011bm_HostExtCorr.SED'),
    ('v19-2011bm', '1.0', 'SN Ic', 'V19_SN2011bm_noHostExtCorr.SED'),
    ('v19-2011dh-corr', '1.0', 'SN IIb', 'V19_SN2011dh_HostExtCorr.SED'),
    ('v19-2011dh', '1.0', 'SN IIb', 'V19_SN2011dh_noHostExtCorr.SED'),
    ('v19-2011ei-corr', '1.0', 'SN IIb', 'V19_SN2011ei_HostExtCorr.SED'),
    ('v19-2011ei', '1.0', 'SN IIb', 'V19_SN2011ei_noHostExtCorr.SED'),
    ('v19-2011fu-corr', '1.0', 'SN IIb', 'V19_SN2011fu_HostExtCorr.SED'),
    ('v19-2011fu', '1.0', 'SN IIb', 'V19_SN2011fu_noHostExtCorr.SED'),
    ('v19-2011hs-corr', '1.0', 'SN IIb', 'V19_SN2011hs_HostExtCorr.SED'),
    ('v19-2011hs', '1.0', 'SN IIb', 'V19_SN2011hs_noHostExtCorr.SED'),
    ('v19-2011ht-corr', '1.0', 'SN IIn', 'V19_SN2011ht_HostExtCorr.SED'),
    ('v19-2011ht', '1.0', 'SN IIn', 'V19_SN2011ht_noHostExtCorr.SED'),
    ('v19-2012A-corr', '1.0', 'SN II', 'V19_SN2012A_HostExtCorr.SED'),
    ('v19-2012A', '1.0', 'SN II', 'V19_SN2012A_noHostExtCorr.SED'),
    ('v19-2012ap-corr', '1.0', 'SN Ic-BL', 'V19_SN2012ap_HostExtCorr.SED'),
    ('v19-2012ap', '1.0', 'SN Ic-BL', 'V19_SN2012ap_noHostExtCorr.SED'),
    ('v19-2012au-corr', '1.0', 'SN Ib', 'V19_SN2012au_HostExtCorr.SED'),
    ('v19-2012au', '1.0', 'SN Ib', 'V19_SN2012au_noHostExtCorr.SED'),
    ('v19-2012aw-corr', '1.0', 'SN II', 'V19_SN2012aw_HostExtCorr.SED'),
    ('v19-2012aw', '1.0', 'SN II', 'V19_SN2012aw_noHostExtCorr.SED'),
    ('v19-2013ab-corr', '1.0', 'SN II', 'V19_SN2013ab_HostExtCorr.SED'),
    ('v19-2013ab', '1.0', 'SN II', 'V19_SN2013ab_noHostExtCorr.SED'),
    ('v19-2013am-corr', '1.0', 'SN II', 'V19_SN2013am_HostExtCorr.SED'),
    ('v19-2013am', '1.0', 'SN II', 'V19_SN2013am_noHostExtCorr.SED'),
    ('v19-2013by-corr', '1.0', 'SN II', 'V19_SN2013by_HostExtCorr.SED'),
    ('v19-2013by', '1.0', 'SN II', 'V19_SN2013by_noHostExtCorr.SED'),
    ('v19-2013df-corr', '1.0', 'SN IIb', 'V19_SN2013df_HostExtCorr.SED'),
    ('v19-2013df', '1.0', 'SN IIb', 'V19_SN2013df_noHostExtCorr.SED'),
    ('v19-2013ej-corr', '1.0', 'SN II', 'V19_SN2013ej_HostExtCorr.SED'),
    ('v19-2013ej', '1.0', 'SN II', 'V19_SN2013ej_noHostExtCorr.SED'),
    ('v19-2013fs-corr', '1.0', 'SN II', 'V19_SN2013fs_HostExtCorr.SED'),
    ('v19-2013fs', '1.0', 'SN II', 'V19_SN2013fs_noHostExtCorr.SED'),
    ('v19-2013ge-corr', '1.0', 'SN Ic', 'V19_SN2013ge_HostExtCorr.SED'),
    ('v19-2013ge', '1.0', 'SN Ic', 'V19_SN2013ge_noHostExtCorr.SED'),
    ('v19-2014G-corr', '1.0', 'SN II', 'V19_SN2014G_HostExtCorr.SED'),
    ('v19-2014G', '1.0', 'SN II', 'V19_SN2014G_noHostExtCorr.SED'),
    ('v19-2016X-corr', '1.0', 'SN II', 'V19_SN2016X_HostExtCorr.SED'),
    ('v19-2016X', '1.0', 'SN II', 'V19_SN2016X_noHostExtCorr.SED'),
    ('v19-2016bkv-corr', '1.0', 'SN II', 'V19_SN2016bkv_HostExtCorr.SED'),
    ('v19-2016bkv', '1.0', 'SN II', 'V19_SN2016bkv_noHostExtCorr.SED'),
    ('v19-2016gkg-corr', '1.0', 'SN IIb', 'V19_SN2016gkg_HostExtCorr.SED'),
    ('v19-2016gkg', '1.0', 'SN IIb', 'V19_SN2016gkg_noHostExtCorr.SED'),
    ('v19-iptf13bvn-corr', '1.0', 'SN Ib', 'V19_iPTF13bvn_HostExtCorr.SED'),
    ('v19-iptf13bvn', '1.0', 'SN Ib', 'V19_iPTF13bvn_noHostExtCorr.SED')
]

note = "Templates from Vincenzi et al. 19. Each template is extended in the " \
    "ultraviolet (1600AA) and in the near infrared (10000AA). Each template " \
    "can be used in its original version (v19-sn-name) or in its host dust " \
    "extinction corrected version (v19-sn-name-corr)."

for name, vrs, sntype, fn in V19_CC_models:
    relpath = 'models/vincenzi/' + fn
    meta = {'subclass': '`~sncosmo.TimeSeriesSource`',
            'type': sntype,
            'ref': ('Vincenzi2019',
                    'Vincenzi et al. 2019 '
                    '<https://arxiv.org/abs/1908.05228>'),
            'note': note,
            'url': 'https://github.com/maria-vincenzi/PyCoCo_templates'}
    _SOURCES.register_loader(name, load_timeseries_ascii,
                             args=(relpath,), version=vrs, meta=meta)

# Pop III CC SN models from D.Whalen et al. 2013.
meta = {'type': 'PopIII',
        'subclass': '`~sncosmo.TimeSeriesSource`',
        'reference': ('Whalen13',
                      'Whalen et al. 2013 '
                      '<http://adsabs.harvard.edu/abs/2013ApJ...768...95W>'),
        'note': "private communication (D.Whalen, May 2014)."}
for name, fn in [('whalen-z15b', 'popIII-z15B.sed.restframe10pc.dat'),
                 ('whalen-z15d', 'popIII-z15D.sed.restframe10pc.dat'),
                 ('whalen-z15g', 'popIII-z15G.sed.restframe10pc.dat'),
                 ('whalen-z25b', 'popIII-z25B.sed.restframe10pc.dat'),
                 ('whalen-z25d', 'popIII-z25D.sed.restframe10pc.dat'),
                 ('whalen-z25g', 'popIII-z25G.sed.restframe10pc.dat'),
                 ('whalen-z40b', 'popIII-z40B.sed.restframe10pc.dat'),
                 ('whalen-z40g', 'popIII-z40G.sed.restframe10pc.dat')]:
    relpath = 'models/whalen/' + fn
    _SOURCES.register_loader(name, load_timeseries_ascii,
                             args=(relpath, True), version='1.0', meta=meta)


# MLCS2k2
def load_mlcs2k2(relpath, name=None, version=None):
    abspath = DATADIR.abspath(relpath)
    return MLCS2k2Source(abspath, name=name, version=version)


meta = {'type': 'SN Ia',
        'subclass': '`~sncosmo.MLCS2k2Source`',
        'reference': ('Jha07',
                      'Jha, Riess and Kirshner 2007 '
                      '<http://adsabs.harvard.edu/abs/2007ApJ...659..122J>'),
        'note': 'In MLCS2k2 language, this version corresponds to '
        '"MLCS2k2 v0.07 rv19-early-smix vectors"'}
_SOURCES.register_loader('mlcs2k2', load_mlcs2k2,
                         args=('models/mlcs2k2/mlcs2k2.modelflux.v1.0.fits',),
                         version='1.0', meta=meta)


# SNEMO
def load_snemo(relpath, name=None, version=None):
    abspath = DATADIR.abspath(relpath)
    return SNEMOSource(abspath, name=name, version=version)


for name, file, ver in [('snemo2', 'snemo2_ev.dat', '1.0'),
                        ('snemo7', 'snemo7_ev.dat', '1.0'),
                        ('snemo15', 'snemo15_ev.dat', '1.0')]:

    meta = {'type': 'SN Ia', 'subclass': '`~sncosmo.SNEMOSource`',
            'url': 'https://snfactory.lbl.gov/snemo/',
            'reference': ('Saunders18',
                          'Saunders et al. 2018 '
                          '<https://arxiv.org/abs/1810.09476>')}

    _SOURCES.register_loader(name, load_snemo,
                             args=['models/snemo/'+file],
                             version=ver, meta=meta)


# SUGAR models
def load_sugarmodel(relpath, name=None, version=None):
    abspath = DATADIR.abspath(relpath, isdir=True)
    return SUGARSource(abspath, name=name, version=version)


for name, files, ver in [('sugar', 'sugar', '1.0')]:

    meta = {'type': 'SN Ia', 'subclass': '`~sncosmo.SUGARSource`',
            'url': 'http://supernovae.in2p3.fr/sugar_template/',
            'reference': ('Leget20',
                          'Leget et al. 2020 '
                          '<https://doi.org/10.1051/0004-6361/201834954>')}

    _SOURCES.register_loader(name, load_sugarmodel,
                             args=['models/sugar/'+files],
                             version=ver, meta=meta)

# =============================================================================
# MagSystems


def load_ab(name=None):
    return ABMagSystem(name=name)


def load_spectral_magsys_fits(relpath, name=None):
    abspath = DATADIR.abspath(relpath)
    hdulist = fits.open(abspath)
    dispersion = hdulist[1].data['WAVELENGTH']
    flux_density = hdulist[1].data['FLUX']
    hdulist.close()
    refspectrum = SpectrumModel(dispersion, flux_density,
                                unit=(u.erg / u.s / u.cm**2 / u.AA),
                                wave_unit=u.AA)

    return SpectralMagSystem(refspectrum, name=name)


def load_csp(name=None):
    # Values transcribed from
    # http://csp.obs.carnegiescience.edu/data/filters
    # on 13 April 2017
    return CompositeMagSystem(bands={'cspu': ('bd17', 10.519),
                                     'cspg': ('bd17', 9.644),
                                     'cspr': ('bd17', 9.352),
                                     'cspi': ('bd17', 9.250),
                                     'cspb': ('vega', 0.030),
                                     'cspv3014': ('vega', 0.0096),
                                     'cspv3009': ('vega', 0.0096),
                                     'cspv9844': ('vega', 0.0096),
                                     'cspys': ('vega', 0.),
                                     'cspjs': ('vega', 0.),
                                     'csphs': ('vega', 0.),
                                     'cspk': ('vega', 0.),
                                     'cspyd': ('vega', 0.),
                                     'cspjd': ('vega', 0.),
                                     'csphd': ('vega', 0.)},
                              name=name)


def load_ab_b12(name=None):
    # offsets are in the sense (mag_SDSS - mag_AB) = offset
    # -> for example: a source with AB mag = 0. will have SDSS mag = 0.06791
    bands = {'sdssu': ('ab', 0.06791),
             'sdssg': ('ab', -0.02028),
             'sdssr': ('ab', -0.00493),
             'sdssi': ('ab', -0.01780),
             'sdssz': ('ab', -0.01015)}

    # add aliases for above
    for letter in 'ugriz':
        bands['sdss::' + letter] = bands['sdss' + letter]

    families = {'megacampsf::u': ('ab', 0.0),
                'megacampsf::g': ('ab', 0.0),
                'megacampsf::r': ('ab', 0.0),
                'megacampsf::i': ('ab', 0.0),
                'megacampsf::z': ('ab', 0.0),
                'megacampsf::y': ('ab', 0.0)}

    return CompositeMagSystem(bands=bands, families=families, name=name)


def load_jla1(name=None):
    """JLA1 magnitude system based on BD+17 STIS v003 spectrum"""

    base = load_spectral_magsys_fits("spectra/bd_17d4708_stisnic_003.fits")
    bands = {'standard::u': (base, 9.724),
             'standard::b': (base, 9.907),
             'standard::v': (base, 9.464),
             'standard::r': (base, 9.166),
             'standard::i': (base, 8.846),
             'keplercam::us': (base, 9.724),
             'keplercam::b': (base, 9.8803),
             'keplercam::v': (base, 9.4722),
             'keplercam::r': (base, 9.3523),
             'keplercam::i': (base, 9.2542),
             '4shooter2::us': (base, 9.724),
             '4shooter2::b': (base, 9.8744),
             '4shooter2::v': (base, 9.4789),
             '4shooter2::r': (base, 9.1554),
             '4shooter2::i': (base, 8.8506),
             'swope2::u': (base, 10.514),
             'swope2::g': (base, 9.64406),
             'swope2::r': (base, 9.3516),
             'swope2::i': (base, 9.25),
             'swope2::b': (base, 9.876433),
             'swope2::v': (base, 9.476626),
             'swope2::v1': (base, 9.471276),
             'swope2::v2': (base, 9.477482)}

    return CompositeMagSystem(bands=bands, name=name)


_MAGSYSTEMS.register_loader(
    'jla1', load_jla1,
    meta={'subclass': '`~sncosmo.CompositeMagSystem`',
          'url': 'http://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html',
          'description': ('JLA1 magnitude system based on BD+17 '
                          'STIS v003 spectrum')})

_MAGSYSTEMS.alias('vega2', 'jla1')

# AB
_MAGSYSTEMS.register_loader(
    'ab', load_ab,
    meta={'subclass': '`~sncosmo.ABMagSystem`',
          'description': 'Source of 3631 Jy has magnitude 0 in all bands'})

# Vega, BD17
website = 'ftp://ftp.stsci.edu/cdbs/calspec/'
subclass = '`~sncosmo.SpectralMagSystem`'
vega_desc = 'Vega (alpha lyrae) has magnitude 0 in all bands.'
bd17_desc = 'BD+17d4708 has magnitude 0 in all bands.'
for name, fn, desc in [('vega', 'alpha_lyr_stis_007.fits', vega_desc),
                       ('bd17', 'bd_17d4708_stisnic_005.fits', bd17_desc)]:
    _MAGSYSTEMS.register_loader(name, load_spectral_magsys_fits,
                                args=('spectra/' + fn,),
                                meta={'subclass': subclass, 'url': website,
                                      'description': desc})

# CSP
_MAGSYSTEMS.register_loader(
    'csp', load_csp,
    meta={'subclass': '`~sncosmo.CompositeMagSystem`',
          'url': 'http://csp.obs.carnegiescience.edu/data/filters',
          'description': 'Carnegie Supernova Project magnitude system.'})


# ab_b12
_MAGSYSTEMS.register_loader(
    'ab-b12', load_ab_b12,
    meta={'subclass': '`~sncosmo.CompositeMagSystem`',
          'url': 'http://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html',
          'description': 'Betoule et al (2012) calibration of SDSS system.'})

_MAGSYSTEMS.alias('ab_b12', 'ab-b12')
_MAGSYSTEMS.alias('vegahst', 'vega')
