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

# Dictionary of urls
urls = None


def get_url(name, version=None):
    """Get the URL for the remote data associated with name, version"""
    global urls

    # Only download the URL look up table once.
    if urls is None:
        from six.moves.urllib.request import urlopen
        import json
        f = urlopen("http://sncosmo.github.io/data/urls.json")
        urls = json.load(f)
        f.close()

    key = name if (version is None) else "{0}_v{1}".format(name, version)
    return urls[key]


def load_bandpass(pkg_data_name, name=None):
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.AA, name=name)


def load_jwst_bandpass(pkg_data_name, name=None):
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.micron, name=name)


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


def load_timeseries_ascii(name=None, version=None):
    remote_url = get_url(name, version)
    with get_readable_fileobj(remote_url, cache=True) as f:
        phases, wavelengths, flux = io.read_griddata_ascii(f)
    return TimeSeriesSource(phases, wavelengths, flux,
                            name=name, version=version)


def load_timeseries_fits(name=None, version=None):
    remote_url = get_url(name, version)
    fn = download_file(remote_url, cache=True)
    phase, wave, flux = io.read_griddata_fits(fn)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version)


def load_timeseries_fits_local(pkg_data_name, name=None, version=None):
    fname = get_pkg_data_filename(pkg_data_name)
    phase, wave, flux = io.read_griddata_fits(fname)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version)


def load_salt2model(topdir, name=None, version=None):
    remote_url = get_url(name, version)
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


def load_2011fe(name=None, version=None):

    remote_url = get_url(name, version)

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


def load_ab(name=None):
    return ABMagSystem(name=name)


def load_spectral_magsys_fits(name=None):
    remote_url = get_url(name)
    fn = download_file(remote_url, cache=True)
    hdulist = fits.open(fn)
    dispersion = hdulist[1].data['WAVELENGTH']
    flux_density = hdulist[1].data['FLUX']
    hdulist.close()
    refspectrum = Spectrum(dispersion, flux_density,
                           unit=(u.erg / u.s / u.cm**2 / u.AA), wave_unit=u.AA)
    return SpectralMagSystem(refspectrum, name=name)

# =============================================================================
# Bandpasses

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


# HST NICMOS
nicmos_meta = {'filterset': 'nicmos2',
               'dataurl': 'http://www.stsci.edu/hst/',
               'retrieved': '05 Aug 2014',
               'description': 'Hubble Space Telescope NICMOS2 filters'}
registry.register_loader(Bandpass, 'nicf110w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_nicmos_nic2_f110w.dat'],
                         meta=nicmos_meta)
registry.register_loader(Bandpass, 'nicf160w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_nicmos_nic2_f160w.dat'],
                         meta=nicmos_meta)


# HST WFC3-IR
wfc3ir_meta = {'filterset': 'wfc3-ir',
               'dataurl': 'http://www.stsci.edu/hst/wfc3/ins_performance/'
                          'throughputs/Throughput_Tables',
               'retrieved': '05 Aug 2014',
               'description': 'Hubble Space Telescope WFC3 IR filters'}
registry.register_loader(Bandpass, 'f098m', load_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f098m.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f105w', load_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f105w.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f110w', load_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f110w.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f125w', load_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f125w.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f127m', load_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f127m.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f139m', load_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f139m.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f140w', load_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f140w.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f153m', load_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f153m.dat'],
                         meta=wfc3ir_meta)
registry.register_loader(Bandpass, 'f160w', load_bandpass,
                         args=['data/bandpasses/hst/hst_wfc3_ir_f160w.dat'],
                         meta=wfc3ir_meta)


# HST WFC3-UVIS
wfc3uvis_meta = {'filterset': 'wfc3-uvis',
                 'dataurl': 'http://www.stsci.edu/hst/wfc3/ins_performance/'
                            'throughputs/Throughput_Tables',
                 'retrieved': '05 Aug 2014',
                 'description': 'Hubble Space Telescope WFC3 UVIS filters'}
registry.register_loader(Bandpass, 'f218w',   load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f218w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f225w',   load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f225w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f275w',   load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f275w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f300x',   load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f300x.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f336w',   load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f336w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f350lp',  load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f350lp.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f390w',   load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f390w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f689m',   load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f689m.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f763m',   load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f763m.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f845m',   load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f845m.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'f438w',   load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f438w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf475w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f475w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf555w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f555w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf606w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f606w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf625w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f625w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf775w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f775w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf814w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f814w.dat'],
                         meta=wfc3uvis_meta)
registry.register_loader(Bandpass, 'uvf850lp', load_bandpass, args=[
                         'data/bandpasses/hst/hst_wfc3_uvis_f850lp.dat'],
                         meta=wfc3uvis_meta)


# HST ACS
acs_meta = {'filterset': 'acs',
            'dataurl': 'http://www.stsci.edu/hst/acs/analysis/throughputs',
            'retrieved': '05 Aug 2014',
            'description': 'Hubble Space Telescope ACS WFC filters'}
registry.register_loader(Bandpass, 'f435w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f435w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f475w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f475w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f555w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f555w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f606w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f606w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f625w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f625w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f775w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f775w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f814w', load_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f814w.dat'],
                         meta=acs_meta)
registry.register_loader(Bandpass, 'f850lp', load_bandpass, args=[
                         'data/bandpasses/hst/hst_acs_wfc_f850lp.dat'],
                         meta=acs_meta)


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


# JWST MIRI (idealized tophat functions)
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


# Kepler
kepler_meta = {
    'filterset': 'kepler',
    'retrieved': '14 Jan 2015',
    'description': 'Bandpass for the Kepler spacecraft',
    'dataurl': 'http://keplergo.arc.nasa.gov/CalibrationResponse.shtml'}
registry.register_loader(Bandpass, 'kepler', load_bandpass,
                         args=['data/bandpasses/kepler.dat'],
                         meta=kepler_meta)


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
for name, ver, sntype, ref in [('nugent-sn1a', '1.2', 'SN Ia', n02ref),
                               ('nugent-sn91t', '1.1', 'SN Ia', s04ref),
                               ('nugent-sn91bg', '1.1', 'SN Ia', n02ref),
                               ('nugent-sn1bc', '1.1', 'SN Ib/c', l05ref),
                               ('nugent-hyper', '1.2', 'SN Ib/c', l05ref),
                               ('nugent-sn2p', '1.2', 'SN IIP', g99ref),
                               ('nugent-sn2l', '1.2', 'SN IIL', g99ref),
                               ('nugent-sn2n', '2.1', 'SN IIn', g99ref)]:
    registry.register_loader(Source, name, load_timeseries_ascii,
                             version=ver,
                             meta={'url': website, 'type': sntype,
                                   'subclass': subclass, 'reference': ref})


# Sako et al 2011 models
ref = ('S11', 'Sako et al. 2011 '
       '<http://adsabs.harvard.edu/abs/2011ApJ...738..162S>')
website = 'http://sdssdp62.fnal.gov/sdsssn/SNANA-PUBLIC/'
subclass = '`~sncosmo.TimeSeriesSource`'
note = "extracted from SNANA's SNDATA_ROOT on 29 March 2013."

for name, sntype in [('s11-2004hx', 'SN IIL/P'),
                     ('s11-2005lc', 'SN IIP'),
                     ('s11-2005hl', 'SN Ib'),
                     ('s11-2005hm', 'SN Ib'),
                     ('s11-2005gi', 'SN IIP'),
                     ('s11-2006fo', 'SN Ic'),
                     ('s11-2006jo', 'SN Ib'),
                     ('s11-2006jl', 'SN IIP')]:
    meta = {'url': website, 'type': sntype, 'subclass': subclass,
            'reference': ref, 'note': note}
    registry.register_loader(Source, name, load_timeseries_ascii,
                             version='1.0', meta=meta)


# Hsiao models
meta = {'url': 'http://csp.obs.carnegiescience.edu/data/snpy',
        'type': 'SN Ia',
        'subclass': '`~sncosmo.TimeSeriesSource`',
        'reference': ('H07', 'Hsiao et al. 2007 '
                      '<http://adsabs.harvard.edu/abs/2007ApJ...663.1187H>'),
        'note': 'extracted from the SNooPy package on 21 Dec 2012.'}
for version in ['1.0', '2.0', '3.0']:
    registry.register_loader(Source, 'hsiao', load_timeseries_fits,
                             version=version, meta=meta)

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
    registry.register_loader(Source, 'salt2', load_salt2model, args=[topdir],
                             version=ver, meta=meta)


# SALT2 extended
meta = {'type': 'SN Ia',
        'subclass': '`~sncosmo.SALT2Source`',
        'url': 'http://sdssdp62.fnal.gov/sdsssn/SNANA-PUBLIC/',
        'note': "extracted from SNANA's SNDATA_ROOT on 15 August 2013."}
registry.register_loader(Source, 'salt2-extended', load_salt2model,
                         args=['salt2_extended'], version='1.0',
                         meta=meta)


# 2011fe
meta = {'type': 'SN Ia',
        'subclass': '`~sncosmo.TimeSeriesSource`',
        'url': 'http://snfactory.lbl.gov/snf/data',
        'reference': ('P13', 'Pereira et al. 2013 '
                      '<http://adsabs.harvard.edu/abs/2013A%26A...554A..27P>')}
registry.register_loader(Source, '2011fe', load_2011fe, version='1.0',
                         meta=meta)


# SNANA CC SN models
url = 'http://das.sdss2.org/ge/sample/sdsssn/SNANA-PUBLIC/'
subclass = '`~sncosmo.TimeSeriesSource`'
ref = ('SNANA', 'Kessler et al. 2009 '
       '<http://adsabs.harvard.edu/abs/2009PASP..121.1028K>')
note = "extracted from SNANA's SNDATA_ROOT on 5 August 2014."
for name, sntype in [('snana-2004fe', 'SN Ic'),
                     ('snana-2004gq', 'SN Ic'),
                     ('snana-sdss004012', 'SN Ic'),
                     ('snana-sdss013195', 'SN Ic'),  # PSNID
                     ('snana-sdss014475', 'SN Ic'),
                     ('snana-sdss015475', 'SN Ic'),
                     ('snana-sdss017548', 'SN Ic'),
                     ('snana-04D1la', 'SN Ic'),
                     ('snana-04D4jv', 'SN Ic'),
                     ('snana-2004gv', 'SN Ib'),
                     ('snana-2006ep', 'SN Ib'),
                     ('snana-2007Y', 'SN Ib'),
                     ('snana-sdss000020', 'SN Ib'),
                     ('snana-sdss002744', 'SN Ib'),  # PSNID
                     ('snana-sdss014492', 'SN Ib'),  # PSNID
                     ('snana-sdss019323', 'SN Ib'),
                     ('snana-sdss000018', 'SN IIP'),  # PSNID
                     ('snana-sdss003818', 'SN IIP'),  # PSNID
                     ('snana-sdss013376', 'SN IIP'),
                     ('snana-sdss014450', 'SN IIP'),
                     ('snana-sdss014599', 'SN IIP'),  # PSNID
                     ('snana-sdss015031', 'SN IIP'),
                     ('snana-sdss015320', 'SN IIP'),
                     ('snana-sdss015339', 'SN IIP'),
                     ('snana-sdss017564', 'SN IIP'),
                     ('snana-sdss017862', 'SN IIP'),
                     ('snana-sdss018109', 'SN IIP'),
                     ('snana-sdss018297', 'SN IIP'),
                     ('snana-sdss018408', 'SN IIP'),
                     ('snana-sdss018441', 'SN IIP'),
                     ('snana-sdss018457', 'SN IIP'),
                     ('snana-sdss018590', 'SN IIP'),
                     ('snana-sdss018596', 'SN IIP'),
                     ('snana-sdss018700', 'SN IIP'),
                     ('snana-sdss018713', 'SN IIP'),
                     ('snana-sdss018734', 'SN IIP'),
                     ('snana-sdss018793', 'SN IIP'),
                     ('snana-sdss018834', 'SN IIP'),
                     ('snana-sdss018892', 'SN IIP'),
                     ('snana-sdss020038', 'SN IIP'),
                     ('snana-sdss012842', 'SN IIn'),
                     ('snana-sdss013449', 'SN IIn')]:
    meta = {'url': url, 'subclass': subclass, 'type': sntype, 'ref': ref,
            'note': note}
    registry.register_loader(Source, name, load_timeseries_ascii,
                             version='1.0', meta=meta)


# Pop III CC SN models from D.Whalen et al. 2013.
meta = {'type': 'PopIII',
        'subclass': '`~sncosmo.TimeSeriesSource`',
        'reference': ('Whalen13',
                      'Whalen et al. 2013 '
                      '<http://adsabs.harvard.edu/abs/2013ApJ...768...95W>'),
        'note': "private communication (D.Whalen, May 2014)."}
for name in ['whalen-z15B', 'whalen-z15D', 'whalen-z15G', 'whalen-z25B',
             'whalen-z25D', 'whalen-z25G', 'whalen-z40B', 'whalen-z40G']:
    registry.register_loader(Source, name, load_timeseries_ascii,
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
for name, desc in [('vega', vega_desc), ('bd17', bd17_desc)]:
    registry.register_loader(MagSystem, name, load_spectral_magsys_fits,
                             meta={'subclass': subclass, 'url': website,
                                   'description': desc})
