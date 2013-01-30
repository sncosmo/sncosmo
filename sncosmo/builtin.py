# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Orchestration of built-in data.

This module keeps track of what built-in bandpasses, spectra and models exist,
where that data lives, and how to read it (using functions in `io` module).
It provides factory functions for generating Bandpass, Spectrum and
TransientModel objects from the built-in data."""

import os

import numpy as np
from astropy.io import ascii
from astropy.io import fits

from .core import Bandpass, Spectrum
from . import models
from . import io

bandpass_dir = 'bandpasses'
model_dir = 'models'
spectrum_dir = 'spectra'

builtin_models = {
    'hsiao': {'versions': ['3.0', '2.0', '1.0'],
              'files': ['Hsiao_SED.fits', 'Hsiao_SED_V2.fits',
                        'Hsiao_SED_V3.fits']},
    'salt2': {'versions': ['2.0', '1.1']},
    'nugent-sn1a': {'versions': ['1.2'], 'files': ['sn1a_flux.v1.2.dat']},
    'nugent-sn91t': {'versions': ['1.1'], 'files': ['sn91t_flux.v1.1.dat']},
    'nugent-sn91bg': {'versions': ['1.1'], 'files': ['sn91bg_flux.v1.1.dat']},
    'nugent-sn1bc': {'versions': ['1.1'], 'files': ['sn1bc_flux.v1.1.dat']},
    'nugent-hyper': {'versions': ['1.2'], 'files': ['hyper_flux.v1.2.dat']},
    'nugent-sn2p': {'versions': ['1.2'], 'files': ['sn2p_flux.v1.2.dat']},
    'nugent-sn2l': {'versions': ['1.2'], 'files': ['sn2l_flux.v1.2.dat']},
    'nugent-sn2n': {'versions': ['2.1'], 'files': ['sn2n_flux.v2.1.dat']}}

builtin_bandpasses = {
    'Bessell::B': {'file': os.path.join('Bessell', 'B.dat')},
    'DECam::DESg': {'file': os.path.join('DECam', 'DESg.dat')},
    'DECam::DESr': {'file': os.path.join('DECam', 'DESr.dat')},
    'DECam::DESi': {'file': os.path.join('DECam', 'DESi.dat')},
    'DECam::DESz': {'file': os.path.join('DECam', 'DESz.dat')},
    'DECam::DESy': {'file': os.path.join('DECam', 'DESy.dat')}}

builtin_spectra = ['vega', 'ab']



def datadir():
    """Return full path to root level of data directory.

    Raises ``ValueError`` if SNCOSMO_DATA environment variable not set.
    """

    if 'SNCOSMO_DATA' not in os.environ:
        raise ValueError('Data directory (SNCOSMO_DATA) not defined.')
    return os.environ['SNCOSMO_DATA']

    # If data is included in path.
    #datadir = path.join(path.dirname(path.abspath(__file__)), 'data')

def _checkfile(filename):
    if not os.path.exists(filename):
        raise ValueError('Built-in data file {} not found.'.format(filename))


def bandpass(name):
    """Create and return a built-in Bandpass."""

    if name not in builtin_bandpasses:
        raise ValueError('{} is not a built-in bandpass.')

    filename = os.path.join(datadir(), bandpass_dir,
                            builtin_bandpasses[name]['file'])
    _checkfile(filename)

    t = ascii.read(filename, names=['wl', 'trans'])
    return Bandpass(t['wl'], t['trans'])


def spectrum(name):
    """Create and return a built-in Spectrum."""

    if name not in builtin_spectra:
        raise ValueError("'{}' is not a built-in spectrum.".format(name))

    spectrumdir = os.path.join(datadir(), spectrum_dir)

    if name == 'ab':
        filename = os.path.join(spectrumdir, 'AB.dat')
        _checkfile(filename)
        t = ascii.read(filename, names=['wl', 'f'])
        return Spectrum(t['wl'], t['f'])

    if name == 'vega':
        filename = os.path.join(spectrumdir, 'alpha_lyr_stis_003.fits')
        _checkfile(filename)
        hdulist = fits.open(filename)
        tbdata = hdulist[1].data
        return Spectrum(tbdata.field('WAVELENGTH'),
                        tbdata.field('FLUX'))


def model(name, version='latest', modelpath=None):
    """Create and return a `models.Transient` instance.

    Parameters
    ----------
    name : str
        Name of model
    version : str, optional
        Version string, e.g., ``'1.1'``. Default is ``'latest'`` which
        corresponds to the highest available version number.

    Returns
    -------
    model : `models.Transient` subclass
        

    Notes
    -----

    Available models:

    ============= ======= ======= ========== ============= ======= ======== ===========
    Name          Version Type    Class      References    Website Data URL Retrieved
    ============= ======= ======= ========== ============= ======= ======== ===========
    salt2         1.1     SN Ia   SALT2      [#0]_ [#1]_   `a`_    `b`_     20 Dec 2012
    salt2         2.0     SN Ia   SALT2                    `a`_    `c`_     20 Dec 2012
    hsiao         1.0     SN Ia   TimeSeries [#2]_         `d`_    [#3]_    23 Nov 2012
    hsiao         2.0     SN Ia   TimeSeries               `d`_    [#3]_    23 Nov 2012
    hsiao         3.0     SN Ia   TimeSeries               `d`_    [#3]_    23 Nov 2012
    nugent-sn1a   1.2     SN Ia   TimeSeries               `e`_    `f`_     20 Dec 2012
    nugent-sn91bg 1.1     SN Ia   TimeSeries               `e`_    `f`_     20 Dec 2012
    nugent-sn91t  1.1     SN Ia   TimeSeries               `e`_    `f`_     20 Dec 2012
    nugent-sn1bc  1.1     SN Ib/c TimeSeries               `e`_    `f`_     20 Dec 2012
    nugent-hyper  1.2     SN Ib/c TimeSeries               `e`_    `f`_     20 Dec 2012
    nugent-sn2p   1.2     SN IIp  TimeSeries               `e`_    `f`_     20 Dec 2012
    nugent-sn2l   1.2     SN IIL  TimeSeries               `e`_    `f`_     20 Dec 2012
    nugent-sn2n   2.1     SN IIn  TimeSeries               `e`_    `f`_     20 Dec 2012
    ============= ======= ======= ========== ============= ======= ======== ===========



    .. [#0] Guy et al. 2007 `[ADS] <http://adsabs.harvard.edu/abs/2007A%26A...466...11G>`_
    .. [#1] Guy et al. 2010 `[ADS] <http://adsabs.harvard.edu/abs/2010A%26A...523A...7G>`_
    .. [#2] Hsiao et al. 2007 `[ADS] <http://adsabs.harvard.edu/abs/2007ApJ...663.1187H>`_

    .. [#3] Extracted from the ``snpy`` package source.

    .. _`a`: http://supernovae.in2p3.fr/~guy/salt/download_templates.html
    .. _`b`: http://supernovae.in2p3.fr/~guy/salt/download/salt2_model_data-1-1.tar.gz
    .. _`c`: http://supernovae.in2p3.fr/~guy/salt/download/salt2_model_data-2-0.tar.gz
    .. _`d`: http://csp.obs.carnegiescience.edu/data/snpy
    .. _`e`: http://supernova.lbl.gov/~nugent/nugent_templates.html
    .. _`f`: http://supernova.lbl.gov/~nugent/templates/


"""

    if modelpath is None:
        if 'SNCOSMO_DATA' in os.environ:
            modelpath = os.path.join(os.environ['SNCOSMO_DATA'], 'models')
        else:
            raise ValueError('If modelpath is not given, environment '
                             'variable SNCOSMO_DATA must be defined.')

    if (name not in builtin_models):
        raise ValueError('No model {}. Available models:\n'.format(name) +
                         ' '.join(builtin_models.keys()))
        
    if (version not in builtin_models[name]['versions'] and
        version != 'latest'):
        raise ValueError("No version '{}' for that model. \n".format(version) +
                         "Available versions: " +
                         ", ".join(builtin_models[name]['versions']))
    
    if version == 'latest':
        version = sorted(builtin_models[name]['versions'])[-1]

    modeldir = os.path.join(datadir(), model_dir, name)

    if name == 'salt2':
        modeldir = os.path.join(modeldir, version)
        files = {'M0': 'salt2_template_0.dat',
                 'M1': 'salt2_template_1.dat',
                 'V00': 'salt2_spec_variance_0.dat',
                 'V01': 'salt2_spec_variance_1.dat',
                 'V11': 'salt2_spec_covariance_01.dat',
                 'errscale': None}
        if version == '1.1':
            files['errscale'] = 'salt2_spec_dispersion_scaling.dat'

        # Check that files exist.
        for key in files:
            if files[key] is None: continue
            _checkfile(os.path.join(modeldir, files[key]))

        return models.SALT2(modeldir,
                            m0file=files['M0'],
                            m1file=files['M1'],
                            v00file=files['V00'],
                            v11file=files['V11'],
                            v01file=files['V01'],
                            errscalefile=files['errscale'])


    elif name == 'hsiao':
        i = builtin_models['hsiao']['versions'].index(version)
        filename = os.path.join(modeldir, builtin_models['hsiao']['files'][i])
        _checkfile(filename)
        hdulist = fits.open(filename)
        hdr = hdulist[0].header

        # Make the phases array
        phase_lo = hdr['CRVAL2'] + (1 - hdr['CRPIX2']) * hdr['CDELT2']
        phase_hi = hdr['CRVAL2'] + (hdr['NAXIS2'] - hdr['CRPIX2']) * hdr['CDELT2']
        phases = np.linspace(phase_lo, phase_hi, hdr['NAXIS2'])

        # Make the wavelengths array
        wavelength_lo = hdr['CRVAL1'] + (1 - hdr['CRPIX1']) * hdr['CDELT1']
        wavelength_hi = hdr['CRVAL1'] + (hdr['NAXIS1'] - hdr['CRPIX1']) * hdr['CDELT1']
        wavelengths = np.linspace(wavelength_lo, wavelength_hi, hdr['NAXIS1'])

        flux = hdulist[0].data

        return models.TimeSeries(phases, wavelengths, flux)

    elif name[0:6] == 'nugent':

        i = builtin_models[name]['versions'].index(version)
        filename = os.path.join(modeldir, builtin_models[name]['files'][i])
        _checkfile(filename)
        phases, wavelengths, flux = io.read_griddata_txt(filename)
        return modes.TimeSeries(phases, wavelengths, flux)
