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

bandpass_dir = 'bandpasses'
model_dir = 'models'
spectrum_dir = 'spectra'

builtin_models = {
    'hsiao': {'class': models.SpectralTimeSeries,
              'versions': ['1', '2', '3']},
    'salt2': {'class': models.SALT2,
              'versions': ['2.2.0', '2.1.0']}}

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


def model(name, version='latest', modelpath=None):
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
        
    if not (version in _builtin_models[name]['versions'] or
            version == 'latest'):
        raise ValueError('No version {} for that model.'.format(version) +
                         'Available versions are:\n' +
                         builtin_models_str())
    
    if version == 'latest':
        version = sorted(_builtin_models[name]['versions'])[-1]

    modeldir = os.path.join(modelpath, name, version)
    return _builtin_models[name]['class'](modeldir)



#def builtin_models_str():
#    """Return string listing built-in models"""
#    s = ""
#    for name, model in _builtin_models.iteritems():
#        for version in model['versions']:
#            s.append("{} {}\n".format(name, version))
#    return s
