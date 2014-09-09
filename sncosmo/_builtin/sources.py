# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Reader functions for initializing built-in data."""

import string
import tarfile
import warnings
from os.path import join

import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.utils.data import (download_file, get_pkg_data_filename,
                                get_readable_fileobj, REMOTE_TIMEOUT)

# Note on REMOTE_TIMEOUT: we have to explicitly import REMOTE_TIMEOUT
# and feed it to the download_file() function. This is because of a bug in
# download_file: REMOTE_TIMEOUT is evaluated when the module is
# loaded rather than when the function is called. This means that
# changing the value of REMOTE_TIMEOUT has no effect on download_file().

from .. import registry
from .. import Source, TimeSeriesSource, SALT2Source
from .. import io


def load_timeseries_ascii(remote_url, name=None, version=None):
    with get_readable_fileobj(remote_url, cache=True) as f:
        phases, wavelengths, flux = io.read_griddata_ascii(f)
    return TimeSeriesSource(phases, wavelengths, flux,
                            name=name, version=version)

def load_timeseries_ascii_local(pkg_data_name, name=None, version=None):
    fname = get_pkg_data_filename(pkg_data_name)
    phase, wave, flux = io.read_griddata_ascii(fname)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version)

def load_timeseries_fits(remote_url, name=None, version=None):
    fn = download_file(remote_url, cache=True, timeout=REMOTE_TIMEOUT())
    phase, wave, flux = io.read_griddata_fits(fn)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version)


def load_timeseries_fits_local(pkg_data_name, name=None, version=None):
    fname = get_pkg_data_filename(pkg_data_name)
    phase, wave, flux = io.read_griddata_fits(fname)
    return TimeSeriesSource(phase, wave, flux, name=name, version=version)


# ------------------------------------------------------------------------
# Nugent models

baseurl = 'http://supernova.lbl.gov/~nugent/templates/'
website = 'http://supernova.lbl.gov/~nugent/nugent_templates.html'
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
                         args=['../data/models/Hsiao_SED_V3_subsampled.fits'],
                         version='3.0', meta=meta)

# -----------------------------------------------------------------------
# SALT2 models

def load_salt2model(remote_url, topdir, name=None, version=None):
    fn = download_file(remote_url, cache=True, timeout=REMOTE_TIMEOUT())
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

    tarfname = download_file(remote_url, cache=True, timeout=REMOTE_TIMEOUT())
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
# SNANA CC SN models added by S.Rodney 2014.09.09

meta={'url':'http://das.sdss2.org/ge/sample/sdsssn/SNANA-PUBLIC/',
      'type':'Ic',
      'subclass':'`~sncosmo.TimeSeriesSource`',
      'reference':('SNANA', 'Kessler et al. 2009 '
                   '<http://adsabs.harvard.edu/abs/2009PASP..121.1028K>'),
      'note':"extracted from SNANA's SNDATA_ROOT on 5 August 2014."}

for modelname,sedfile,type in [
    ['Ic.01', 'CSP-2004fe.SED',  'Ic'],
    ['Ic.02', 'CSP-2004gq.SED',  'Ic'],
    ['Ic.03', 'SDSS-004012.SED', 'Ic'],
    ['Ic.04', 'SDSS-013195.SED', 'Ic'],
    ['Ic.05', 'SDSS-014475.SED', 'Ic'],
    ['Ic.06', 'SDSS-015475.SED', 'Ic'],
    ['Ic.07', 'SDSS-017548.SED', 'Ic'],
    ['Ic.08', 'SNLS-04D1la.SED', 'Ic'],
    ['Ic.09', 'SNLS-04D4jv.SED', 'Ic'],
    ['Ib.01', 'CSP-2004gv.SED',  'Ib'],
    ['Ib.02', 'CSP-2006ep.SED',  'Ib'],
    ['Ib.03', 'CSP-2007Y.SED',   'Ib'],
    ['Ib.04', 'SDSS-000020.SED', 'Ib'],
    ['Ib.05', 'SDSS-002744.SED', 'Ib'],
    ['Ib.06', 'SDSS-014492.SED', 'Ib'],
    ['Ib.07', 'SDSS-019323.SED', 'Ib'],
    ['IIP.01','SDSS-000018.SED', 'IIP'],
    ['IIP.02','SDSS-003818.SED', 'IIP'],
    ['IIP.03','SDSS-013376.SED', 'IIP'],
    ['IIP.04','SDSS-014450.SED', 'IIP'],
    ['IIP.05','SDSS-014599.SED', 'IIP'],
    ['IIP.06','SDSS-015031.SED', 'IIP'],
    ['IIP.07','SDSS-015320.SED', 'IIP'],
    ['IIP.08','SDSS-015339.SED', 'IIP'],
    ['IIP.09','SDSS-017564.SED', 'IIP'],
    ['IIP.10','SDSS-017862.SED', 'IIP'],
    ['IIP.11','SDSS-018109.SED', 'IIP'],
    ['IIP.12','SDSS-018297.SED', 'IIP'],
    ['IIP.13','SDSS-018408.SED', 'IIP'],
    ['IIP.14','SDSS-018441.SED', 'IIP'],
    ['IIP.15','SDSS-018457.SED', 'IIP'],
    ['IIP.16','SDSS-018590.SED', 'IIP'],
    ['IIP.17','SDSS-018596.SED', 'IIP'],
    ['IIP.18','SDSS-018700.SED', 'IIP'],
    ['IIP.19','SDSS-018713.SED', 'IIP'],
    ['IIP.20','SDSS-018734.SED', 'IIP'],
    ['IIP.21','SDSS-018793.SED', 'IIP'],
    ['IIP.22','SDSS-018834.SED', 'IIP'],
    ['IIP.23','SDSS-018892.SED', 'IIP'],
    ['IIP.24','SDSS-020038.SED', 'IIP'],
    ['IIn.01','SDSS-012842.SED', 'IIN'],
    ['IIn.02','SDSS-013449.SED', 'IIN'],
    ] :

    meta.update( {'snid':sedfile.split('.')[0]} )
    registry.register_loader(Source, modelname, load_timeseries_ascii_local,
                             args=['../data/models/ccsn/'+sedfile],
                             version='1.0',
                             meta=meta )

# --------------------------------------------------------------------------
# Pop III CC SN models from D.Whalen et al. 2013.
# added by S.Rodney 2014.09.09

for mod in ['z15B','z15D','z15G','z25B','z25D','z25G','z40B','z40G'] :
    registry.register_loader(Source,mod,load_timeseries_ascii_local,
                             args=['../data/models/whalen/popIII-%s.sed.restframe10pc.dat'%mod],
                             version='1.0',
                             meta={'snid':mod,'type':'PopIII',
                                   'subclass':'`~sncosmo.TimeSeriesSource`',
                                   'reference':('Whalen13',
                                                'Whalen et al. 2013 '
                                                '<http://adsabs.harvard.edu/abs/2013ApJ...768...95W>'),
                                   'note':"private communication (D.Whalen to S.Rodney), May 2014."})


# --------------------------------------------------------------------------
# Generate docstring

lines = [
    '',
    '  '.join([20*'=', 7*'=', 8*'=', 27*'=', 14*'=', 7*'=', 7*'=']),
    '{0:20}  {1:7}  {2:8}  {3:27}  {4:14}  {5:7}  {6:50}'.format(
        'Name', 'Version', 'Type', 'Subclass', 'Reference', 'Website', 'Notes')
    ]
lines.append(lines[1])

urlnums = {}
allnotes = []
allrefs = []
for m in registry.get_loaders_metadata(Source):

    reflink = ''
    urllink = ''
    notelink = ''

    if 'note' in m:
        if m['note'] not in allnotes:
            allnotes.append(m['note'])
        notenum = allnotes.index(m['note'])
        notelink = '[{0}]_'.format(notenum + 1)

    if 'reference' in m:
        reflink = '[{0}]_'.format(m['reference'][0])
        if m['reference'] not in allrefs:
            allrefs.append(m['reference'])

    if 'url' in m:
        url = m['url']
        if url not in urlnums:
            if len(urlnums) == 0:
                urlnums[url] = 0
            else:
                urlnums[url] = max(urlnums.values()) + 1
        urllink = '`{0}`_'.format(string.letters[urlnums[url]])

    lines.append("{0!r:20}  {1!r:7}  {2:8}  {3:27}  {4:14}  {5:7}  {6:50}"
                 .format(m['name'], m['version'], m['type'], m['subclass'],
                         reflink, urllink, notelink))

lines.extend([lines[1], ''])
for refkey, ref in allrefs:
    lines.append('.. [{0}] `{1}`__'.format(refkey, ref))
lines.append('')
for url, urlnum in urlnums.iteritems():
    lines.append('.. _`{0}`: {1}'.format(string.letters[urlnum], url))
lines.append('')
for i, note in enumerate(allnotes):
    lines.append('.. [{0}] {1}'.format(i + 1, note))
lines.append('')
__doc__ = '\n'.join(lines)

# Clean up the module namespace.
del lines
del urlnums
del allrefs
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
