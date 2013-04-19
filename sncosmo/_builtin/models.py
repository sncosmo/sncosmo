# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Reader functions for initializing built-in data."""

import string
import tarfile
from os.path import join

import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.utils.data import (download_file, get_pkg_data_filename,
                                get_readable_fileobj)

from .. import registry
from .. import Model, TimeSeriesModel, SALT2Model
from .. import utils

# ------------------------------------------------------------------------
# Bandflux relative errors

def set_bandfluxerror_sn1a(model):
    dmin, dmax = model.disp(modelframe=True)[[0, -1]]
    disp = np.linspace(dmin, dmax, (dmax-dmin)/100. + 1)
    phase = model.times(modelframe=True)
    relphase = np.abs(phase - model.refphase)
    relerror = np.empty(relphase.shape, dtype=np.float)
    idx = relphase < 20.
    relerror[idx] = 0.08 + 0.04 * relphase[idx] / 20.
    idx = np.invert(idx)
    relerror[idx] = 0.12 + 0.08 * (relphase[idx] - 20.) / 60.
    relerror = relerror[:, np.newaxis]
    relerror = relerror * (((disp < 4000.) | (disp > 8300.)) + 1.)
    model.set_bandflux_relative_error(phase, disp, relerror)

def set_bandfluxerror_sncc(model):
    dmin, dmax = model.disp(modelframe=True)[[0, -1]]
    disp = np.linspace(dmin, dmax, (dmax-dmin)/100. + 1)
    phase = model.times(modelframe=True)
    relphase = phase - model.refphase
    relerror = (0.08 + 0.08 * np.abs(relphase) / 60.)[:, np.newaxis]
    relerror = relerror * (((disp < 4000.) | (disp > 8300.)) + 1.)
    model.set_bandflux_relative_error(phase, disp, relerror)

# ------------------------------------------------------------------------
# Nugent models

def load_timeseries_ascii_sn1a(remote_url, name=None, version=None):
    with get_readable_fileobj(remote_url, cache=True) as f:
        phases, wavelengths, flux = utils.read_griddata(f)
    model = TimeSeriesModel(phases, wavelengths, flux,
                            name=name, version=version)
    set_bandfluxerror_sn1a(model)
    return model

def load_timeseries_ascii_sncc(remote_url, name=None, version=None):
    with get_readable_fileobj(remote_url, cache=True) as f:
        phases, wavelengths, flux = utils.read_griddata(f)
    model = TimeSeriesModel(phases, wavelengths, flux,
                            name=name, version=version)
    set_bandfluxerror_sncc(model)
    return model

# ------------------------------------------------------------------------
# Nugent models

nugent_baseurl = 'http://supernova.lbl.gov/~nugent/templates/'
nugent_website = 'http://supernova.lbl.gov/~nugent/nugent_templates.html'
nugent_subclass = '`~sncosmo.TimeSeriesModel`'

registry.register_loader(
    Model, 'nugent-sn1a', load_timeseries_ascii_sn1a, 
    [nugent_baseurl + 'sn1a_flux.v1.2.dat.gz'],
    version='1.2', url=nugent_website, type='SN Ia',
    subclass=nugent_subclass,
    reference=('N02', 'Nugent, Kim & Permutter 2002 '
               '<http://adsabs.harvard.edu/abs/2002PASP..114..803N>'))

registry.register_loader(
    Model, 'nugent-sn91t', load_timeseries_ascii_sn1a, 
    [nugent_baseurl + 'sn91t_flux.v1.1.dat.gz'],
    version='1.1', url=nugent_website, type='SN Ia',
    subclass=nugent_subclass,
    reference=('S04', 'Stern, et al. 2004 '
               '<http://adsabs.harvard.edu/abs/2004ApJ...612..690S>'))

registry.register_loader(
    Model, 'nugent-sn91bg', load_timeseries_ascii_sn1a, 
    [nugent_baseurl + 'sn91bg_flux.v1.1.dat.gz'],
    version='1.1', url=nugent_website, type='SN Ia',
    subclass=nugent_subclass,
    reference=('N02', 'Nugent, Kim & Permutter 2002 '
               '<http://adsabs.harvard.edu/abs/2002PASP..114..803N>'))

registry.register_loader(
    Model, 'nugent-sn1bc', load_timeseries_ascii_sncc, 
    [nugent_baseurl + 'sn1bc_flux.v1.1.dat.gz'],
    version='1.1', url=nugent_website, type='SN Ib/c',
    subclass=nugent_subclass,
    reference=('L05', 'Levan et al. 2005 '
               '<http://adsabs.harvard.edu/abs/2005ApJ...624..880L>'))

registry.register_loader(
    Model, 'nugent-hyper', load_timeseries_ascii_sncc, 
    [nugent_baseurl + 'hyper_flux.v1.2.dat.gz'],
    version='1.2', url=nugent_website, type='SN Ib/c',
    subclass=nugent_subclass,
    reference=('L05', 'Levan et al. 2005 '
               '<http://adsabs.harvard.edu/abs/2005ApJ...624..880L>'))

registry.register_loader(
    Model, 'nugent-sn2p', load_timeseries_ascii_sncc, 
    [nugent_baseurl + 'sn2p_flux.v1.2.dat.gz'],
    version='1.2', url=nugent_website, type='SN IIP',
    subclass=nugent_subclass,
    reference=('G99', 'Gilliland, Nugent & Phillips 1999 '
               '<http://adsabs.harvard.edu/abs/1999ApJ...521...30G>'))

registry.register_loader(
    Model, 'nugent-sn2l', load_timeseries_ascii_sncc, 
    [nugent_baseurl + 'sn2l_flux.v1.2.dat.gz'],
    version='1.2', url=nugent_website, type='SN IIL',
    subclass=nugent_subclass,
    reference=('G99', 'Gilliland, Nugent & Phillips 1999 '
               '<http://adsabs.harvard.edu/abs/1999ApJ...521...30G>'))

registry.register_loader(
    Model, 'nugent-sn2n', load_timeseries_ascii_sncc, 
    [nugent_baseurl + 'sn2n_flux.v2.1.dat.gz'],
    version='2.1', url=nugent_website, type='SN IIn',
    subclass=nugent_subclass,
    reference=('G99', 'Gilliland, Nugent & Phillips 1999 '
               '<http://adsabs.harvard.edu/abs/1999ApJ...521...30G>'))

del nugent_website
del nugent_baseurl
del nugent_subclass

# -----------------------------------------------------------------------
# Sako et al 2011 models

s11_baseurl = 'http://kbarbary.github.com/data/models/'
s11_ref = ('S11', 'Sako et al. 2011 '
           '<http://adsabs.harvard.edu/abs/2011ApJ...738..162S>')
s11_website = 'http://sdssdp62.fnal.gov/sdsssn/SNANA-PUBLIC/'
s11_subclass = '`~sncosmo.TimeSeriesModel`'
s11_note = "extracted from SNANA's SNDATA_ROOT on 29 March 2013."

registry.register_loader(
    Model, 's11-2004hx', load_timeseries_ascii_sncc,
    [s11_baseurl + 'S11_SDSS-000018.SED'], version='1.0', url=s11_website,
    type='SN IIL/P', subclass=s11_subclass, reference=s11_ref, note=s11_note)

registry.register_loader(
    Model, 's11-2005lc', load_timeseries_ascii_sncc,
    [s11_baseurl + 'S11_SDSS-001472.SED'], version='1.0', url=s11_website,
    type='SN IIP', subclass=s11_subclass, reference=s11_ref, note=s11_note)

registry.register_loader(
    Model, 's11-2005hl', load_timeseries_ascii_sncc,
    [s11_baseurl + 'S11_SDSS-002000.SED'], version='1.0', url=s11_website,
    type='SN Ib', subclass=s11_subclass, reference=s11_ref, note=s11_note)

registry.register_loader(
    Model, 's11-2005hm', load_timeseries_ascii_sncc,
    [s11_baseurl + 'S11_SDSS-002744.SED'], version='1.0', url=s11_website,
    type='SN Ib', subclass=s11_subclass, reference=s11_ref, note=s11_note)

registry.register_loader(
    Model, 's11-2005gi', load_timeseries_ascii_sncc,
    [s11_baseurl + 'S11_SDSS-003818.SED'], version='1.0', url=s11_website,
    type='SN IIP', subclass=s11_subclass, reference=s11_ref, note=s11_note)

registry.register_loader(
    Model, 's11-2006fo', load_timeseries_ascii_sncc,
    [s11_baseurl + 'S11_SDSS-013195.SED'], version='1.0', url=s11_website,
    type='SN Ic', subclass=s11_subclass, reference=s11_ref, note=s11_note)

registry.register_loader(
    Model, 's11-2006jo', load_timeseries_ascii_sncc,
    [s11_baseurl + 'S11_SDSS-014492.SED'], version='1.0', url=s11_website,
    type='SN Ib', subclass=s11_subclass, reference=s11_ref, note=s11_note)

registry.register_loader(
    Model, 's11-2006jl', load_timeseries_ascii_sncc,
    [s11_baseurl + 'S11_SDSS-014599.SED'], version='1.0', url=s11_website,
    type='SN IIP', subclass=s11_subclass, reference=s11_ref, note=s11_note)

del s11_baseurl
del s11_ref
del s11_website
del s11_subclass
del s11_note

# -----------------------------------------------------------------------
# Hsiao models

def load_timeseries_fits(remote_url, name=None, version=None):
    fn = download_file(remote_url, cache=True)
    hdulist = fits.open(fn)
    w = wcs.WCS(hdulist[0].header)
    flux_density = hdulist[0].data
    hdulist.close()
    
    ny, nx = flux_density.shape
    xcoords = np.arange(nx)
    ycoords = np.arange(ny)

    coords = np.swapaxes([xcoords, np.zeros(nx)], 0, 1)
    dispersion = w.wcs_pix2world(coords, 0)[:,0]

    coords = np.swapaxes([np.zeros(ny), ycoords], 0, 1)
    phase = w.wcs_pix2world(coords, 0)[:,1]
    model = TimeSeriesModel(phase, dispersion, flux_density, name=name,
                            version=version)
    set_bandfluxerror_sn1a(model)
    return model

hsiao_baseurl = 'http://kbarbary.github.com/data/models/'
hsiao_website = 'http://csp.obs.carnegiescience.edu/data/snpy'
hsiao_subclass = '`~sncosmo.TimeSeriesModel`'
hsiao_ref = ('H07', 'Hsiao et al. 2007 <http://adsabs.harvard.edu/abs/'
             '2007ApJ...663.1187H>')
hsiao_note = 'extracted from the SNooPy package on 21 Dec 2012.'

registry.register_loader(
    Model, 'hsiao', load_timeseries_fits, [hsiao_baseurl + 'Hsiao_SED.fits'],
    version='1.0', url=hsiao_website, type='SN Ia', subclass=hsiao_subclass,
    reference=hsiao_ref, note=hsiao_note)
registry.register_loader(
    Model, 'hsiao', load_timeseries_fits,[hsiao_baseurl + 'Hsiao_SED_V2.fits'],
    version='2.0', url=hsiao_website, type='SN Ia', subclass=hsiao_subclass,
    reference=hsiao_ref, note=hsiao_note)
registry.register_loader(
    Model, 'hsiao', load_timeseries_fits,[hsiao_baseurl + 'Hsiao_SED_V3.fits'],
    version='3.0', url=hsiao_website, type='SN Ia', subclass=hsiao_subclass,
    reference=hsiao_ref, note=hsiao_note)

del hsiao_baseurl
del hsiao_website
del hsiao_subclass
del hsiao_ref
del hsiao_note

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

    model = SALT2Model(
        m0file=t.extractfile(join(topdir,'salt2_template_0.dat')),
        m1file=t.extractfile(join(topdir,'salt2_template_1.dat')),
        v00file=t.extractfile(join(topdir,'salt2_spec_variance_0.dat')),
        v11file=t.extractfile(join(topdir,'salt2_spec_variance_1.dat')),
        v01file=t.extractfile(join(topdir,'salt2_spec_covariance_01.dat')),
        errscalefile=errscalefile, name=name, version=version)
    t.close()
    set_bandfluxerror_sn1a(model)
    return model

salt2_baseurl = 'http://supernovae.in2p3.fr/~guy/salt/download/'
salt2_website = 'http://supernovae.in2p3.fr/~guy/salt/download_templates.html'
salt2_reference = ('G07', 'Guy et al. 2007 '
                   '<http://adsabs.harvard.edu/abs/2007A%26A...466...11G>')
registry.register_loader(
    Model, 'salt2', load_salt2model,
    [salt2_baseurl + 'salt2_model_data-1-1.tar.gz', 'salt2-1-1'],
    version='1.1', type='SN Ia', subclass='`~sncosmo.SALT2Model`', 
    url=salt2_website, reference=salt2_reference)
registry.register_loader(
    Model, 'salt2', load_salt2model,
    [salt2_baseurl + 'salt2_model_data-2-0.tar.gz', 'salt2-2-0'],
    version='2.0', type='SN Ia', subclass='`~sncosmo.SALT2Model`', 
    url=salt2_website, reference=salt2_reference)


# --------------------------------------------------------------------------
# Generate docstring

lines = [
    '',
    '  '.join([16*'=', 7*'=', 8*'=', 27*'=', 14*'=', 7*'=', 7*'=']),
    '{:16}  {:7}  {:8}  {:27}  {:14}  {:7}  {:50}'.format(
        'Name', 'Version', 'Type', 'Subclass', 'Reference', 'Website', 'Notes')
    ]
lines.append(lines[1])

urlnums = {}
allnotes = []
allrefs = []
for m in registry.get_loaders_metadata(Model):

    reflink = ''
    urllink = ''
    notelink = ''

    if 'note' in m:
        if m['note'] not in allnotes: allnotes.append(m['note'])
        notenum = allnotes.index(m['note'])
        notelink = '[{}]_'.format(notenum + 1)

    if 'reference' in m:
        reflink = '[{}]_'.format(m['reference'][0])
        if m['reference'] not in allrefs:
            allrefs.append(m['reference'])

    if 'url' in m:
        url = m['url']
        if url not in urlnums:
            if len(urlnums) == 0: urlnums[url] = 0
            else: urlnums[url] = max(urlnums.values()) + 1
        urllink = '`{}`_'.format(string.letters[urlnums[url]])

    lines.append("{0!r:16}  {1!r:7}  {2:8}  {3:27}  {4:14}  {5:7}  {6:50}"
                 .format(m['name'], m['version'], m['type'], m['subclass'],
                         reflink, urllink, notelink))

lines.extend([lines[1], ''])
for refkey, ref in allrefs:
    lines.append('.. [{}] `{}`__'.format(refkey, ref))
lines.append('')
for url, urlnum in urlnums.iteritems():
    lines.append('.. _`{}`: {}'.format(string.letters[urlnum], url))
lines.append('')
for i, note in enumerate(allnotes):
    lines.append('.. [{}] {}'.format(i + 1, note))
lines.append('')
__doc__ = '\n'.join(lines)

del lines
del urlnums
del allrefs
