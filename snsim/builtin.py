# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Orchestration of built-in data.

This module keeps track of what built-in bandpasses, spectra and models exist,
where that data lives, and how to read it (using functions in snsim.io).
It provides factory functions for generating Bandpass, Spectrum and
TransientModel objects from the built-in data."""

import os

from astropy.io import ascii

from .core import Bandpass, Spectrum
from . import models
from . import io

bandpass_dir = 'bandpasses'
model_dir = 'models'
spectrum_dir = 'spectra'

builtin_models = {
    'hsiao': {'versions': ['1.0', '2.0', '3.0']},
    'salt2': {'versions': ['2.2.0', '2.1.0']}}

builtin_bandpasses = {
    'DECam::DESg': {'file': os.path.join('DECam', 'DESg.dat')},
    'DECam::DESr': {'file': os.path.join('DECam', 'DESr.dat')},
    'DECam::DESi': {'file': os.path.join('DECam', 'DESi.dat')},
    'DECam::DESz': {'file': os.path.join('DECam', 'DESz.dat')},
    'DECam::DESy': {'file': os.path.join('DECam', 'DESy.dat')}}

builtin_spectra = {
    'vega': {'file': 'AB.dat'},
    'ab': {'file': 'vega.dat'}}


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
    result = Bandpass(t['wl'], t['trans'], copy=True)
    del t
    return result


def spectrum(name):
    """Create and return a built-in Spectrum."""

    if name not in builtin_spectra:
        raise ValueError('{} is not a built-in spectrum.')

    filename = os.path.join(datadir(), spectrum_dir,
                            builtin_spectra[name]['file'])
    _checkfile(filename)

    t = ascii.read(filename, names=['wl', 'f'])
    result = Spectrum(t['wl'], t['f'], copy=True)
    del t
    return result


def model(name, version='latest'):
    """Create and return a Transient class instance."""

    if modelpath is None:
        if 'SNSIM_DATA' in os.environ:
            modelpath = os.path.join(os.environ['SNSIM_DATA'], 'models')
        else:
            raise ValueError('If modelpath is not given, environment '
                             'variable SNSIM_DATA must be defined.')

    if (name not in _builtin_models):
        raise ValueError('No model {}. Available models:\n'.format(name) +
                         ' '.join(_builtin_models.keys()))
        
    if (version not in _builtin_models[name]['versions'] and
        version != 'latest'):
        raise ValueError('No version {} for that model.'.format(version) +
                         'Available versions are:' +
                         ','.join(_builtin_models[name]['versions']))
    
    if version == 'latest':
        version = sorted(_builtin_models[name]['versions'])[-1]

    modeldir = os.path.join(datadir(), model_dir, name, version)

    if name == 'salt2':
        files = {'M0': 'salt2_template_0.dat',
                 'M1': 'salt2_template_1.dat',
                 'V00': 'salt2_spec_variance_0.dat',
                 'V01': 'salt2_spec_variance_1.dat',
                 'V11': 'salt2_spec_covariance_01.dat',
                 'errorscale': 'salt2_spec_dispersion_scaling.dat'}

        phases = None
        wavelengths = None
        components = {}

        phases, wavelengths, M0 = io.read_griddata_txt(
            os.path.join(modeldir, 'salt2_template_0.dat'))

        # Read grid data from each component file.
        for key in files:
            full_filename = os.path.join(modeldir, files[key])
            if not os.path.exists(full_filename):
                if key == 'errorscale':
                    components[key] = None
                    continue
                else:
                    raise ValueError("File not found: '{}'"
                                     .format(full_filename))

            x0, x1, y = io.read_griddata_txt(full_filename)

            # The first component determines the phase and wavelength
            if phases is None:
                phases = x0
                wavelengths = x1

            # Others must match
            elif (x0 != phases or x1 != wavelengths):
                raise ValueError('Model components must have matching phases'
                                 ' and wavelengths.')

            components[key] = y

        # Return the SALT2 model
        return models.SALT2(phases, wavelengths, components['M0'], 
                            components['M1'], components['V00'],
                            components['V11'], components['V01'],
                            errorscale=components['errorscale'])

    elif name == 'hsiao':
        print "not implemented yet"
        exit()
