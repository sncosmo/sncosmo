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

import numpy as np
from astropy import wcs, units as u
from astropy.io import ascii, fits
from astropy.config import ConfigurationItem, get_cache_dir
from astropy.extern import six
from astropy.utils import OrderedDict
from astropy.utils.data import get_pkg_data_filename

from . import registry
from . import io
from .utils import download_file, download_dir
from .models import Source, TimeSeriesSource, SALT2Source, MLCS2k2Source
from .spectral import (Bandpass, read_bandpass, Spectrum, MagSystem,
                       SpectralMagSystem, ABMagSystem)
from . import conf

# This module is only imported for its side effects.
__all__ = []

# Dictionary of urls, used by get_url()
urls = None


def get_url(name, version=None):
    """Get the URL for the remote data associated with name, version"""
    global urls

    # Only download the URL look up table once.
    if urls is None:
        from six.moves.urllib.request import urlopen
        import json
        f = urlopen("http://sncosmo.github.io/data/urls.json")
        reader = codecs.getreader("utf-8")
        urls = json.load(reader(f))
        f.close()

    key = name if (version is None) else "{0}_v{1}".format(name, version)

    return urls[key]


def get_data_dir():
    """Return the full path to the data directory, ensuring that it exists.

    If the 'data_dir' configuration parameter is set, checks that it exists
    and returns it (doesn't automatically create it).

    Otherwise, uses `(astropy cache dir)/sncosmo` (automatically created if
    it doesn't exist)."""
    if conf.data_dir is not None:
        if os.path.isdir(conf.data_dir):
            return conf.data_dir
        else:
            raise RuntimeError(
                "data directory {0!r} not an existing directory"
                .format(conf.data_dir))
    else:
        data_dir = join(get_cache_dir(), "sncosmo")
        if not os.path.isdir(data_dir):
            if os.path.exists(data_dir):
                raise RuntimeError("{0} not a directory".format(data_dir))
            else:
                os.mkdir(data_dir)
        return data_dir


def get_abspath(relpath, name, version=None):
    """Return the absolute path to a sncosmo data file, ensuring that
    it exists (file will be downloaded if needed).

    Parameters
    ----------
    relpath : str
        Relative path; data directory will be appended.
    name : str
        Name of built-in, used to look up URL if needed.
    version : str
        Version of built-in, used to look up URL if needed.
    """

    abspath = join(get_data_dir(), relpath)

    if not os.path.exists(abspath):
        url = get_url(name, version)

        # If it's a tar file, download and unpack a directory.
        if url.endswith(".tar.gz") or url.endswith(".tar"):
            dirname = os.path.dirname(abspath)
            download_dir(url, dirname)

            # ensure that tarfile unpacked into the expected directory
            if not os.path.exists(abspath):
                raise RuntimeError("Tarfile not unpacked into expected "
                                   "subdirectory. Please file an issue.")

        # Otherwise, its a single file.
        else:
            download_file(url, abspath)

    return abspath


def load_bandpass_angstroms(pkg_data_name, name=None):
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.AA, name=name)


def load_bandpass_microns(pkg_data_name, name=None):
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.micron, name=name)


def load_bandpass_bessell(pkg_data_name, name=None):
    """Bessell bandpasses have (1/energy) transmission units."""
    fname = get_pkg_data_filename(pkg_data_name)
    band = read_bandpass(fname, wave_unit=u.AA, trans_unit=u.erg**-1,
                         name=name)

    # We happen to know that Bessell bandpasses in file are arbitrarily
    # scaled to have a peak of 1 photon / erg. Rescale here to a peak of
    # 1 (unitless transmission) to be more similar to other bandpasses.
    band.trans /= np.max(band.trans)

    return band


def tophat_bandpass(ctr, width, name=None):
    """Create a tophat Bandpass centered at `ctr` with width `width` (both
    in microns) sampled at 100 intervals."""

    nintervals = 100  # intervals between wmin and wmax
    interval = width / nintervals  # interval of each sample
    wmin = ctr - width / 2. - interval / 2.
    wmax = ctr + width / 2. + interval / 2.
    wave = np.linspace(wmin, wmax, nintervals + 1)
    trans = np.ones_like(wave)
    trans[[0, -1]] = 0.
    return Bandpass(wave, trans, wave_unit=u.micron, name=name)


def load_timeseries_ascii(relpath, zero_before=False, name=None, version=None):
    abspath = get_abspath(relpath, name, version=version)
    phase, wave, flux = io.read_griddata_ascii(abspath)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version,
                            zero_before=zero_before)


def load_timeseries_fits(relpath, name=None, version=None):
    abspath = get_abspath(relpath, name, version=version)
    phase, wave, flux = io.read_griddata_fits(abspath)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version)


def load_timeseries_fits_local(pkg_data_name, name=None, version=None):
    fname = get_pkg_data_filename(pkg_data_name)
    phase, wave, flux = io.read_griddata_fits(fname)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version)


def load_salt2model(relpath, name=None, version=None):
    abspath = get_abspath(relpath, name, version=version)
    return SALT2Source(modeldir=abspath, name=name, version=version)


def load_2011fe(relpath, name=None, version=None):

    # filter warnings about RADESYS keyword in files
    warnings.filterwarnings('ignore', category=wcs.FITSFixedWarning,
                            append=True)

    abspath = get_abspath(relpath, name, version=version)

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


def load_ab(name=None):
    return ABMagSystem(name=name)


def load_spectral_magsys_fits(relpath, name=None):
    abspath = get_abspath(relpath, name)
    hdulist = fits.open(abspath)
    dispersion = hdulist[1].data['WAVELENGTH']
    flux_density = hdulist[1].data['FLUX']
    hdulist.close()
    refspectrum = Spectrum(dispersion, flux_density,
                           unit=(u.erg / u.s / u.cm**2 / u.AA), wave_unit=u.AA)
    return SpectralMagSystem(refspectrum, name=name)

# =============================================================================
# Bandpasses

bessell_meta = {
    'filterset': 'bessell',
    'reference': ('B90', '`Bessell 1990 <http://adsabs.harvard.edu/'
                  'abs/1990PASP..102.1181B>`__, Table 2'),
    'description': 'Representation of Johnson-Cousins UBVRI system'}

des_meta = {
    'filterset': 'des',
    'retrieved': '22 March 2013',
    'description': 'Dark Energy Camera grizy filter set at airmass 1.3'}

sdss_meta = {
    'filterset': 'sdss',
    'reference': ('D10', '`Doi et al. 2010 <http://adsabs.harvard.edu/'
                  'abs/2010AJ....139.1628D>`__, Table 4'),
    'description': ('SDSS 2.5m imager at airmass 1.3 (including '
                    'atmosphere), normalized')}

nicmos_meta = {'filterset': 'nicmos2',
               'dataurl': 'http://www.stsci.edu/hst/',
               'retrieved': '05 Aug 2014',
               'description': 'Hubble Space Telescope NICMOS2 filters'}

wfc3ir_meta = {'filterset': 'wfc3-ir',
               'dataurl': 'http://www.stsci.edu/hst/wfc3/ins_performance/'
                          'throughputs/Throughput_Tables',
               'retrieved': '05 Aug 2014',
               'description': 'Hubble Space Telescope WFC3 IR filters'}

wfc3uvis_meta = {'filterset': 'wfc3-uvis',
                 'dataurl': 'http://www.stsci.edu/hst/wfc3/ins_performance/'
                            'throughputs/Throughput_Tables',
                 'retrieved': '05 Aug 2014',
                 'description': 'Hubble Space Telescope WFC3 UVIS filters'}

acs_meta = {'filterset': 'acs',
            'dataurl': 'http://www.stsci.edu/hst/acs/analysis/throughputs',
            'retrieved': '05 Aug 2014',
            'description': 'Hubble Space Telescope ACS WFC filters'}

jwst_nircam_meta = {'filterset': 'jwst-nircam',
                    'dataurl': 'http://www.stsci.edu/jwst/instruments/nircam'
                               '/instrumentdesign/filters',
                    'retrieved': '09 Sep 2014',
                    'description': 'James Webb Space Telescope NIRCAM '
                    'Wide+Medium filters'}

kepler_meta = {
    'filterset': 'kepler',
    'retrieved': '14 Jan 2015',
    'description': 'Bandpass for the Kepler spacecraft',
    'dataurl': 'http://keplergo.arc.nasa.gov/CalibrationResponse.shtml'}

# Bessell bandpasses have transmission in units of (photons / erg)
bands = [('bessellux', 'bessell/bessell_ux.dat', bessell_meta),
         ('bessellb', 'bessell/bessell_b.dat', bessell_meta),
         ('bessellv', 'bessell/bessell_v.dat', bessell_meta),
         ('bessellr', 'bessell/bessell_r.dat', bessell_meta),
         ('besselli', 'bessell/bessell_i.dat', bessell_meta)]
for name, fname, meta in bands:
    registry.register_loader(Bandpass, name, load_bandpass_bessell,
                             args=['data/bandpasses/' + fname],
                             meta=meta)

bands = [('desg', 'des/des_g.dat', des_meta),
         ('desr', 'des/des_r.dat', des_meta),
         ('desi', 'des/des_i.dat', des_meta),
         ('desz', 'des/des_z.dat', des_meta),
         ('desy', 'des/des_y.dat', des_meta),
         ('sdssu', 'sdss/sdss_u.dat', sdss_meta),
         ('sdssg', 'sdss/sdss_g.dat', sdss_meta),
         ('sdssr', 'sdss/sdss_r.dat', sdss_meta),
         ('sdssi', 'sdss/sdss_i.dat', sdss_meta),
         ('sdssz', 'sdss/sdss_z.dat', sdss_meta),
         ('nicf110w', 'hst/hst_nicmos_nic2_f110w.dat', nicmos_meta),
         ('nicf160w', 'hst/hst_nicmos_nic2_f160w.dat', nicmos_meta),
         ('f098m', 'hst/hst_wfc3_ir_f098m.dat', wfc3ir_meta),
         ('f105w', 'hst/hst_wfc3_ir_f105w.dat', wfc3ir_meta),
         ('f110w', 'hst/hst_wfc3_ir_f110w.dat', wfc3ir_meta),
         ('f125w', 'hst/hst_wfc3_ir_f125w.dat', wfc3ir_meta),
         ('f127m', 'hst/hst_wfc3_ir_f127m.dat', wfc3ir_meta),
         ('f139m', 'hst/hst_wfc3_ir_f139m.dat', wfc3ir_meta),
         ('f140w', 'hst/hst_wfc3_ir_f140w.dat', wfc3ir_meta),
         ('f153m', 'hst/hst_wfc3_ir_f153m.dat', wfc3ir_meta),
         ('f160w', 'hst/hst_wfc3_ir_f160w.dat', wfc3ir_meta),
         ('f218w', 'hst/hst_wfc3_uvis_f218w.dat', wfc3uvis_meta),
         ('f225w', 'hst/hst_wfc3_uvis_f225w.dat', wfc3uvis_meta),
         ('f275w', 'hst/hst_wfc3_uvis_f275w.dat', wfc3uvis_meta),
         ('f300x', 'hst/hst_wfc3_uvis_f300x.dat', wfc3uvis_meta),
         ('f336w', 'hst/hst_wfc3_uvis_f336w.dat', wfc3uvis_meta),
         ('f350lp', 'hst/hst_wfc3_uvis_f350lp.dat', wfc3uvis_meta),
         ('f390w', 'hst/hst_wfc3_uvis_f390w.dat', wfc3uvis_meta),
         ('f689m', 'hst/hst_wfc3_uvis_f689m.dat', wfc3uvis_meta),
         ('f763m', 'hst/hst_wfc3_uvis_f763m.dat', wfc3uvis_meta),
         ('f845m', 'hst/hst_wfc3_uvis_f845m.dat', wfc3uvis_meta),
         ('f438w', 'hst/hst_wfc3_uvis_f438w.dat', wfc3uvis_meta),
         ('uvf475w', 'hst/hst_wfc3_uvis_f475w.dat', wfc3uvis_meta),
         ('uvf555w', 'hst/hst_wfc3_uvis_f555w.dat', wfc3uvis_meta),
         ('uvf606w', 'hst/hst_wfc3_uvis_f606w.dat', wfc3uvis_meta),
         ('uvf625w', 'hst/hst_wfc3_uvis_f625w.dat', wfc3uvis_meta),
         ('uvf775w', 'hst/hst_wfc3_uvis_f775w.dat', wfc3uvis_meta),
         ('uvf814w', 'hst/hst_wfc3_uvis_f814w.dat', wfc3uvis_meta),
         ('uvf850lp', 'hst/hst_wfc3_uvis_f850lp.dat', wfc3uvis_meta),
         ('f435w', 'hst/hst_acs_wfc_f435w.dat', acs_meta),
         ('f475w', 'hst/hst_acs_wfc_f475w.dat', acs_meta),
         ('f555w', 'hst/hst_acs_wfc_f555w.dat', acs_meta),
         ('f606w', 'hst/hst_acs_wfc_f606w.dat', acs_meta),
         ('f625w', 'hst/hst_acs_wfc_f625w.dat', acs_meta),
         ('f775w', 'hst/hst_acs_wfc_f775w.dat', acs_meta),
         ('f814w', 'hst/hst_acs_wfc_f814w.dat', acs_meta),
         ('f850lp', 'hst/hst_acs_wfc_f850lp.dat', acs_meta),
         ('kepler', 'kepler/kepler.dat', kepler_meta)]

for name, fname, meta in bands:
    registry.register_loader(Bandpass, name, load_bandpass_angstroms,
                             args=['data/bandpasses/' + fname],
                             meta=meta)

bands = [('f070w', 'jwst/jwst_nircam_f070w.dat', jwst_nircam_meta),
         ('f090w', 'jwst/jwst_nircam_f090w.dat', jwst_nircam_meta),
         ('f115w', 'jwst/jwst_nircam_f115w.dat', jwst_nircam_meta),
         ('f150w', 'jwst/jwst_nircam_f150w.dat', jwst_nircam_meta),
         ('f200w', 'jwst/jwst_nircam_f200w.dat', jwst_nircam_meta),
         ('f277w', 'jwst/jwst_nircam_f277w.dat', jwst_nircam_meta),
         ('f356w', 'jwst/jwst_nircam_f356w.dat', jwst_nircam_meta),
         ('f444w', 'jwst/jwst_nircam_f444w.dat', jwst_nircam_meta),
         ('f140m', 'jwst/jwst_nircam_f140m.dat', jwst_nircam_meta),
         ('f162m', 'jwst/jwst_nircam_f162m.dat', jwst_nircam_meta),
         ('f182m', 'jwst/jwst_nircam_f182m.dat', jwst_nircam_meta),
         ('f210m', 'jwst/jwst_nircam_f210m.dat', jwst_nircam_meta),
         ('f250m', 'jwst/jwst_nircam_f250m.dat', jwst_nircam_meta),
         ('f300m', 'jwst/jwst_nircam_f300m.dat', jwst_nircam_meta),
         ('f335m', 'jwst/jwst_nircam_f335m.dat', jwst_nircam_meta),
         ('f360m', 'jwst/jwst_nircam_f360m.dat', jwst_nircam_meta),
         ('f410m', 'jwst/jwst_nircam_f410m.dat', jwst_nircam_meta),
         ('f430m', 'jwst/jwst_nircam_f430m.dat', jwst_nircam_meta),
         ('f460m', 'jwst/jwst_nircam_f460m.dat', jwst_nircam_meta),
         ('f480m', 'jwst/jwst_nircam_f480m.dat', jwst_nircam_meta)]

for name, fname, meta in bands:
    registry.register_loader(Bandpass, name, load_bandpass_microns,
                             args=['data/bandpasses/' + fname],
                             meta=meta)

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
    registry.register_loader(Bandpass, name, tophat_bandpass,
                             args=[ctr, width], meta=jwst_miri_meta)


# =============================================================================
# Sources

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
    registry.register_loader(Source, name, load_timeseries_ascii,
                             args=[relpath], version=ver,
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
    registry.register_loader(Source, name, load_timeseries_ascii,
                             args=['models/sako/' + fn], version='1.0',
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
    registry.register_loader(Source, 'hsiao', load_timeseries_fits,
                             args=['models/hsiao/' + fn], version=version,
                             meta=meta)

# subsampled version of Hsiao v3.0, for testing purposes.
registry.register_loader(Source, 'hsiao-subsampled',
                         load_timeseries_fits_local,
                         args=['data/models/Hsiao_SED_V3_subsampled.fits'],
                         version='3.0', meta=meta)

# SALT2 models
website = 'http://supernovae.in2p3.fr/salt/doku.php?id=salt_templates'
g07ref = ('G07', 'Guy et al. 2007 '
          '<http://adsabs.harvard.edu/abs/2007A%26A...466...11G>')
g10ref = ('G10', 'Guy et al. 2010 '
          '<http://adsabs.harvard.edu/abs/2010A%26A...523A...7G>')
b14ref = ('B14', 'Betoule et al. 2014 '
          '<http://arxiv.org/abs/1401.4064>')
for topdir, ver, ref in [('salt2', '1.0', g07ref),
                         ('salt2-1-1', '1.1', g07ref),
                         ('salt2-2-0', '2.0', g10ref),
                         ('salt2-4', '2.4', b14ref)]:
    meta = {'type': 'SN Ia', 'subclass': '`~sncosmo.SALT2Source`',
            'url': website, 'reference': ref}
    registry.register_loader(Source, 'salt2', load_salt2model,
                             args=['models/salt2/'+topdir],
                             version=ver, meta=meta)

# SALT2 extended
meta = {'type': 'SN Ia',
        'subclass': '`~sncosmo.SALT2Source`',
        'url': 'http://sdssdp62.fnal.gov/sdsssn/SNANA-PUBLIC/',
        'note': "extracted from SNANA's SNDATA_ROOT on 15 August 2013."}
registry.register_loader(Source, 'salt2-extended', load_salt2model,
                         args=['models/snana/salt2_extended'], version='1.0',
                         meta=meta)

# 2011fe
meta = {'type': 'SN Ia',
        'subclass': '`~sncosmo.TimeSeriesSource`',
        'url': 'http://snfactory.lbl.gov/snf/data',
        'reference': ('P13', 'Pereira et al. 2013 '
                      '<http://adsabs.harvard.edu/abs/2013A%26A...554A..27P>')}
registry.register_loader(Source, 'snf-2011fe', load_2011fe, version='1.0',
                         args=['models/snf/SN2011fe'], meta=meta)


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
    registry.register_loader(Source, name, load_timeseries_ascii,
                             args=[relpath], version='1.0', meta=meta)


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
    registry.register_loader(Source, name, load_timeseries_ascii,
                             args=[relpath, True], version='1.0', meta=meta)


# MLCS2k2
def load_mlcs2k2(relpath, name=None, version=None):
    abspath = get_abspath(relpath, name, version=version)
    return MLCS2k2Source(abspath, name=name, version=version)

meta = {'type': 'SN Ia',
        'subclass': '`~sncosmo.MLCS2k2Source`',
        'reference': ('Jha07',
                      'Jha, Riess and Kirshner 2007 '
                      '<http://adsabs.harvard.edu/abs/2007ApJ...659..122J>'),
        'note': 'In MLCS2k2 language, this version corresponds to '
        '"MLCS2k2 v0.07 rv19-early-smix vectors"'}
registry.register_loader(Source, 'mlcs2k2', load_mlcs2k2,
                         args=['models/mlcs2k2/mlcs2k2.modelflux.fits'],
                         version='1.0', meta=meta)

# =============================================================================
# MagSystems

# AB
registry.register_loader(
    MagSystem, 'ab', load_ab,
    meta={'subclass': '`~sncosmo.ABMagSystem`',
          'description': 'Source of 3631 Jy has magnitude 0 in all bands'})

# Vega, BD17
website = 'ftp://ftp.stsci.edu/cdbs/calspec/'
subclass = '`~sncosmo.SpectralMagSystem`'
vega_desc = 'Vega (alpha lyrae) has magnitude 0 in all bands.'
bd17_desc = 'BD+17d4708 has magnitude 0 in all bands.'
for name, fn, desc in [('vega', 'alpha_lyr_stis_007.fits', vega_desc),
                       ('bd17', 'bd_17d4708_stisnic_005.fits', bd17_desc)]:
    registry.register_loader(MagSystem, name, load_spectral_magsys_fits,
                             args=['spectra/' + fn],
                             meta={'subclass': subclass, 'url': website,
                                   'description': desc})
