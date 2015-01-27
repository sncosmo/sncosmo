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


def load_bandpass_angstroms(pkg_data_name, name=None):
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.AA, name=name)


def load_bandpass_microns(pkg_data_name, name=None):
    fname = get_pkg_data_filename(pkg_data_name)
    return read_bandpass(fname, wave_unit=u.micron, name=name)


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

bands = [('bessellux', 'bessell/bessell_ux.dat', bessell_meta),
         ('bessellb', 'bessell/bessell_b.dat', bessell_meta),
         ('bessellv', 'bessell/bessell_v.dat', bessell_meta),
         ('bessellr', 'bessell/bessell_r.dat', bessell_meta),
         ('besselli', 'bessell/bessell_i.dat', bessell_meta),
         ('desg', 'des/des_g.dat', des_meta),
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
