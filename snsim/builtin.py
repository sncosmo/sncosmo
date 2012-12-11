# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Orchestration of built-in data.

This module keeps track of what built-in bandpasses, spectra and models exist,
where that data lives, and how to read it (using functions in snsim.io).
It provides factory functions for generating Bandpass, Spectrum and
TransientModel objects from the built-in data."""

import os

from astropy.io import ascii
from astropy.io import fits

from .core import Bandpass, Spectrum
from . import models
from . import io

bandpass_dir = 'bandpasses'
model_dir = 'models'
spectrum_dir = 'spectra'

builtin_models = {
    'hsiao': {'versions': ['3.0', '2.0', '1.0']},
    'salt2': {'versions': ['2.0', '1.1']}}

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

    Raises ``ValueError`` if SNSIM_DATA environment variable not set.
    """

    if 'SNSIM_DATA' not in os.environ:
        raise ValueError('Data directory not defined.')
    return os.environ['SNSIM_DATA']

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
    """Create and return a Transient class instance."""

    if modelpath is None:
        if 'SNSIM_DATA' in os.environ:
            modelpath = os.path.join(os.environ['SNSIM_DATA'], 'models')
        else:
            raise ValueError('If modelpath is not given, environment '
                             'variable SNSIM_DATA must be defined.')

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

    modeldir = os.path.join(datadir(), model_dir, name, version)

    if name == 'salt2':
        files = {'M0': 'salt2_template_0.dat',
                 'M1': 'salt2_template_1.dat',
                 'V00': 'salt2_spec_variance_0.dat',
                 'V01': 'salt2_spec_variance_1.dat',
                 'V11': 'salt2_spec_covariance_01.dat',
                 'errscale': None}
        if version == '1.1':
            files['errscale'] = 'salt2_spec_dispersion_scaling.dat'

        # Change to full paths and check that they exist.
        for key in files:
            if files[key] is None: continue
            files[key] = os.path.join(modeldir, files[key])
            _checkfile(files[key])

        return models.SALT2(files['M0'], files['M1'], files['V00'],
                            files['V11'], files['V01'],
                            errscalefile=files['errscale'])


    elif name == 'hsiao':
        print "not implemented yet"
        exit()
