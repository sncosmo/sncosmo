# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Importing this module registers loaders for built-in data structures:

- Sources
- Bandpasses
- MagSystems
"""

import string
import tarfile
import warnings
import os
from os.path import join
import codecs
from collections import OrderedDict

import numpy as np
from astropy import wcs, units as u
from astropy.io import ascii, fits
from astropy.config import ConfigItem, get_cache_dir
from astropy.extern import six
from astropy.utils.data import get_pkg_data_filename

from . import io
from . import snfitio
from .utils import download_file, download_dir, DataMirror
from .models import (Source, TimeSeriesSource, SALT2Source, MLCS2k2Source,
                     _SOURCES)
from .bandpasses import (Bandpass, read_bandpass, _BANDPASSES,
                         _BANDPASS_INTERPOLATORS)
from .spectrum import Spectrum
from .magsystems import (MagSystem, SpectralMagSystem, ABMagSystem,
                         CompositeMagSystem, _MAGSYSTEMS)
from . import conf
from .constants import BANDPASS_TRIM_LEVEL

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

def load_bandpass_angstroms(pkg_data_name, name=None):
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.AA, name=name)


def load_bandpass_microns(pkg_data_name, name=None):
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.micron, name=name)


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


des_meta = {
    'filterset': 'des',
    'retrieved': '22 March 2013',
    'description': 'Dark Energy Camera grizy filter set at airmass 1.3'}
for name, fname in [('desg', 'des/des_g.dat'),
                    ('desr', 'des/des_r.dat'),
                    ('desi', 'des/des_i.dat'),
                    ('desz', 'des/des_z.dat'),
                    ('desy', 'des/des_y.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_angstroms,
                                args=('data/bandpasses/' + fname,),
                                meta=des_meta)


sdss_meta = {
    'filterset': 'sdss',
    'reference': ('D10', '`Doi et al. 2010 <http://adsabs.harvard.edu/'
                  'abs/2010AJ....139.1628D>`__, Table 4'),
    'description': ('SDSS 2.5m imager at airmass 1.3 (including '
                    'atmosphere), normalized')}
for name, fname in [('sdssu', 'sdss/sdss_u.dat'),
                    ('sdssg', 'sdss/sdss_g.dat'),
                    ('sdssr', 'sdss/sdss_r.dat'),
                    ('sdssi', 'sdss/sdss_i.dat'),
                    ('sdssz', 'sdss/sdss_z.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_angstroms,
                                args=('data/bandpasses/' + fname,),
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
                    ('f814w', 'bandpasses/acs-wfc/wfc_F814W.dat'),
                    ('f850lp', 'bandpasses/acs-wfc/wfc_F850LP.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(fname,), meta=acs_meta)


# HST NICMOS NIC2 bandpasses: remote
nicmos_meta = {'filterset': 'nicmos2',
               'dataurl': 'http://www.stsci.edu/hst/',
               'retrieved': '05 Aug 2014',
               'description': 'Hubble Space Telescope NICMOS2 filters'}
for name, fname in [
        ('nicf110w', 'bandpasses/nicmos-nic2/hst_nicmos_nic2_f110w.dat'),
        ('nicf160w', 'bandpasses/nicmos-nic2/hst_nicmos_nic2_f160w.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_remote_aa,
                                args=(fname,), meta=nicmos_meta)


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
                                    args=(fname,), meta=nicmos_meta)


wfc3uvis_meta = {'filterset': 'wfc3-uvis',
                 'dataurl': 'http://www.stsci.edu/hst/wfc3/ins_performance/'
                            'throughputs/Throughput_Tables',
                 'retrieved': '05 Aug 2014',
                 'description': 'Hubble Space Telescope WFC3 UVIS filters'}
for name, fname in [('f218w', 'hst/hst_wfc3_uvis_f218w.dat'),
                    ('f225w', 'hst/hst_wfc3_uvis_f225w.dat'),
                    ('f275w', 'hst/hst_wfc3_uvis_f275w.dat'),
                    ('f300x', 'hst/hst_wfc3_uvis_f300x.dat'),
                    ('f336w', 'hst/hst_wfc3_uvis_f336w.dat'),
                    ('f350lp', 'hst/hst_wfc3_uvis_f350lp.dat'),
                    ('f390w', 'hst/hst_wfc3_uvis_f390w.dat'),
                    ('f689m', 'hst/hst_wfc3_uvis_f689m.dat'),
                    ('f763m', 'hst/hst_wfc3_uvis_f763m.dat'),
                    ('f845m', 'hst/hst_wfc3_uvis_f845m.dat'),
                    ('f438w', 'hst/hst_wfc3_uvis_f438w.dat'),
                    ('uvf475w', 'hst/hst_wfc3_uvis_f475w.dat'),
                    ('uvf555w', 'hst/hst_wfc3_uvis_f555w.dat'),
                    ('uvf606w', 'hst/hst_wfc3_uvis_f606w.dat'),
                    ('uvf625w', 'hst/hst_wfc3_uvis_f625w.dat'),
                    ('uvf775w', 'hst/hst_wfc3_uvis_f775w.dat'),
                    ('uvf814w', 'hst/hst_wfc3_uvis_f814w.dat'),
                    ('uvf850lp', 'hst/hst_wfc3_uvis_f850lp.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_angstroms,
                                args=('data/bandpasses/' + fname,),
                                meta=wfc3uvis_meta)


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
    'retrieved': '6 Nov 2015',
    'description': 'Carnegie Supernova Proj. filts (Swope+DuPont Telescopes)',
    'dataurl': 'http://csp.obs.carnegiescience.edu/data/filters'}
for name, fname in [('cspb',     'csp/B_texas_WLcorr_atm.txt'),
                    ('csphs',    'csp/H_SWO_TAM_scan_atm.dat'),
                    ('csphd',    'csp/H_texas_DUP_atm.dat'),
                    ('cspjs',    'csp/J_SWO_TAM_atm.dat'),
                    ('cspjd',    'csp/J_texas_DUP_atm.dat'),
                    ('cspv3009', 'csp/V_LC3009_texas_WLcorr_atm.txt'),
                    ('cspv3014', 'csp/V_LC3014_texas_WLcorr_atm.txt'),
                    ('cspv9844', 'csp/V_LC9844_texax_WLcorr_atm.txt'),
                    ('cspys',    'csp/Y_SWO_TAM_scan_atm.dat'),
                    ('cspyd',    'csp/Y_texas_DUP_atm.dat'),
                    ('cspg',     'csp/g_texas_WLcorr_atm.txt'),
                    ('cspi',     'csp/i_texas_WLcorr_atm.txt'),
                    ('cspk',     'csp/kfilter'),
                    ('cspr',     'csp/r_texas_WLcorr_atm.txt'),
                    ('cspu',     'csp/u_texas_WLcorr_atm.txt')]:
    _BANDPASSES.register_loader(name, load_bandpass_angstroms,
                                args=('data/bandpasses/' + fname,),
                                meta=csp_meta)


jwst_nircam_meta = {'filterset': 'jwst-nircam',
                    'dataurl': 'http://www.stsci.edu/jwst/instruments/nircam'
                               '/instrumentdesign/filters',
                    'retrieved': '09 Sep 2014',
                    'description': 'James Webb Space Telescope NIRCAM '
                                   'Wide+Medium filters'}
for name, fname in [('f070w', 'jwst/jwst_nircam_f070w.dat'),
                    ('f090w', 'jwst/jwst_nircam_f090w.dat'),
                    ('f115w', 'jwst/jwst_nircam_f115w.dat'),
                    ('f150w', 'jwst/jwst_nircam_f150w.dat'),
                    ('f200w', 'jwst/jwst_nircam_f200w.dat'),
                    ('f277w', 'jwst/jwst_nircam_f277w.dat'),
                    ('f356w', 'jwst/jwst_nircam_f356w.dat'),
                    ('f444w', 'jwst/jwst_nircam_f444w.dat'),
                    ('f140m', 'jwst/jwst_nircam_f140m.dat'),
                    ('f162m', 'jwst/jwst_nircam_f162m.dat'),
                    ('f182m', 'jwst/jwst_nircam_f182m.dat'),
                    ('f210m', 'jwst/jwst_nircam_f210m.dat'),
                    ('f250m', 'jwst/jwst_nircam_f250m.dat'),
                    ('f300m', 'jwst/jwst_nircam_f300m.dat'),
                    ('f335m', 'jwst/jwst_nircam_f335m.dat'),
                    ('f360m', 'jwst/jwst_nircam_f360m.dat'),
                    ('f410m', 'jwst/jwst_nircam_f410m.dat'),
                    ('f430m', 'jwst/jwst_nircam_f430m.dat'),
                    ('f460m', 'jwst/jwst_nircam_f460m.dat'),
                    ('f480m', 'jwst/jwst_nircam_f480m.dat')]:
    _BANDPASSES.register_loader(name, load_bandpass_microns,
                                args=('data/bandpasses/' + fname,),
                                meta=jwst_nircam_meta)


jwst_miri_meta = {'filterset': 'jwst-miri',
                  'dataurl': 'http://www.stsci.edu/jwst/instruments/miri/'
                             'instrumentdesign/filters',
                  'retrieved': '09 Sep 2014',
                  'description': 'James Webb Space Telescope MIRI '
                                 'filters (idealized tophats)'}
for name, ctr, width in [('f560w', 5.6, 1.2),
                         ('f770w', 7.7, 2.2),
                         ('f1000w', 10., 2.),
                         ('f1130w', 11.3, 0.7),
                         ('f1280w', 12.8, 2.4),
                         ('f1500w', 15., 3.),
                         ('f1800w', 18., 3.),
                         ('f2100w', 21., 5.),
                         ('f2550w', 25.5, 4.),
                         ('f1065c', 10.65, 0.53),
                         ('f1140c', 11.4, 0.57),
                         ('f1550c', 15.5, 0.78),
                         ('f2300c', 23., 4.6)]:
    _BANDPASSES.register_loader(name, tophat_bandpass_um,
                                args=(ctr, width), meta=jwst_miri_meta)


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


# =============================================================================
# bandpass interpolators


def load_megacampsf(letter, name=None):
    abspath = DATADIR.abspath('bandpasses/megacampsf', isdir=True)
    return snfitio.read_snfit_bandpass_interpolator(abspath, letter, name=name)

for letter in ('u', 'g', 'r', 'i', 'z', 'y'):
    _BANDPASS_INTERPOLATORS.register_loader('megacampsf::' + letter,
                                            load_megacampsf, args=(letter,))

# =============================================================================
# Sources


def load_timeseries_ascii(relpath, zero_before=False, name=None, version=None):
    abspath = DATADIR.abspath(relpath)
    phase, wave, flux = io.read_griddata_ascii(abspath)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version,
                            zero_before=zero_before)


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

    return TimeSeriesSource(phases, disp, flux,
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

for suffix, ver, sntype, ref in [('sn1a', '1.2', 'SN Ia', n02ref),
                                 ('sn91t', '1.1', 'SN Ia', s04ref),
                                 ('sn91bg', '1.1', 'SN Ia', n02ref),
                                 ('sn1bc', '1.1', 'SN Ib/c', l05ref),
                                 ('hyper', '1.2', 'SN Ib/c', l05ref),
                                 ('sn2p', '1.2', 'SN IIP', g99ref),
                                 ('sn2l', '1.2', 'SN IIL', g99ref),
                                 ('sn2n', '2.1', 'SN IIn', g99ref)]:
    name = "nugent-" + suffix
    relpath = "models/nugent/{0}_flux.v{1}.dat".format(suffix, ver)
    _SOURCES.register_loader(name, load_timeseries_ascii,
                             args=(relpath,), version=ver,
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
b14ref = ('B14', 'Betoule et al. 2014 '
          '<http://arxiv.org/abs/1401.4064>')
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
    refspectrum = Spectrum(dispersion, flux_density,
                           unit=(u.erg / u.s / u.cm**2 / u.AA), wave_unit=u.AA)

    return SpectralMagSystem(refspectrum, name=name)


def load_csp(**kwargs):

    # this file contains the csp zeropoints and standards
    fname = get_pkg_data_filename('data/bandpasses/csp/csp_filter_info.dat')
    data = np.genfromtxt(fname, names=True, dtype=None, skip_header=3)
    bands = data['name']
    refsystems = data['reference_sed']
    offsets = data['natural_mag']

    # In Python 3, convert to native strings (Unicode)
    if six.PY3:
        bands = np.char.decode(bands)
        refsystems = np.char.decode(refsystems)

    return CompositeMagSystem(bands, refsystems, offsets, name='csp')


def load_ab_b12(**kwargs):
    # offsets are in the sense (mag_SDSS - mag_AB) = offset
    # -> for example: a source with AB mag = 0. will have SDSS mag = 0.06791
    bands = ['sdssu', 'sdssg', 'sdssr', 'sdssi', 'sdssz',
             'sdss::u', 'sdss::g', 'sdss::r', 'sdss::i', 'sdss::z']
    standards = 10 * ['ab']
    offsets = 2 * [0.06791, -0.02028, -0.00493, -0.01780, -0.01015]
    return CompositeMagSystem(bands, standards, offsets)


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
