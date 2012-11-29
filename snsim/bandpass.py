
from os import path
import numpy as np
from astropy.io import ascii

bandpass_dir = path.join(path.dirname(path.abspath(__file__)), 'data',
                         'bandpasses')
_builtin_bandpasses = {
    'DECam::DESg': path.join(bandpass_dir, 'DECam', 'DESg.dat'),
    'DECam::DESr': path.join(bandpass_dir, 'DECam', 'DESr.dat'),
    'DECam::DESi': path.join(bandpass_dir, 'DECam', 'DESi.dat'),
    'DECam::DESz': path.join(bandpass_dir, 'DECam', 'DESz.dat'),
    'DECam::DESy': path.join(bandpass_dir, 'DECam', 'DESy.dat')}

class Bandpass(object):
    """A bandpass, e.g., filter. ('filter' is a built-in python function.)

    Parameters
    ----------
    wavelength : list_like
        Wavelength values, in angstroms
    transmission : list_like
        Transmission values.
    copy : bool, optional
        Copy input arrays.
    """

    @classmethod
    def get(cls, bandname):
        """Get a built-in bandpass.

        Parameters
        ----------
        bandname : string

        Returns
        -------
        bandpass : Bandpass object
        """
        if bandname not in _builtin_bandpasses:
            raise ValueError('{} is not a built-in bandpass.')
        
        return cls.read(_builtin_bandpasses[bandname])


    @classmethod
    def read(cls, filename):
        """Read a bandpass from an ASCII file.
    
        Must be a two column table, optionally with one line header.

        Parameters
        ----------
        filename : string

        Returns
        -------
        bandpass : Bandpass object
        """
    
        table = ascii.read(filename, names=['wavelength', 'transmission'])
        bandpass = cls(table['wavelength'], table['transmission'],
                       copy=True)
        del table
        return bandpass


    def __init__(self, wavelength, transmission, copy=False):
        
        wavelength = np.array(wavelength, copy=copy)
        transmission = np.array(transmission, copy=copy)
        if wavelength.shape != transmission.shape:
            raise ValueError('shape of wavelength and transmission must match')
        if wavelength.ndim != 1:
            raise ValueError('only 1-d arrays supported')
        self.wavelength = wavelength
        self.transmission = transmission


    def size(self):
        return self.wavelength.shape[0]


    def range(self):
        return (self.wavelength[0], self.wavelength[-1])
