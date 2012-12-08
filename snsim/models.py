# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Classes to represent spectral models of astronomical transients."""

import abc
import os
import math

import numpy as np

from .utils import GridData

__all__ = ['Transient', 'SpectralTimeSeries', 'SALT2']


class Transient(object):
    """Abstract base class for transient spectral models.

    A model can be anything that has a defined spectrum depending on phase
    and any other number of parameters. For example, in the SALT2 model, the
    phase, x1, and c values uniquely determine a spectrum."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def wavelengths(self):
        """Return wavelength coverage array of the model (as a copy).
        
        Returns
        -------
        wavelengths : numpy.ndarray
            Wavelength values in Angstroms.
        """
        return

    @abc.abstractmethod
    def phases(self):
        """Return array of phases sampled in the model. 
 
        Returns
        -------
        phases : numpy.ndarray
            Phases in days.
        """
        return

    @abc.abstractmethod
    def flux(phase, wavelengths=None):
        """Return flux of the model at the given phase.
        
        Parameters
        ----------
        phase : float
            Rest frame phase in days.
        wavelengths : list_like, optional
            Wavelength values at which to evaluate flux. If `None` use
            native model wavelengths.

        Returns
        -------
        flux : numpy.ndarray
            Flux values in ??.
        """
        return


class SpectralTimeSeries(Transient):
    """A spectrum for each phase, optionally with error.

    Parameters
    ----------
    phase : 1-d array
    wavelength : 1-d array
    flux : 2-d array of shape (len(phase), len(wavelength))
    fluxerr : 2-d array, optional

    """

    def __init__(self, phases, wavelengths, flux, fluxerr=None):
        
        self._phases = phases
        self._wavelengths = wavelengths
        self._model = GridData(phases, wavelengths, flux)
        self._modelerr = None
        if fluxerr is not None:
            self._modelerr = GridData(phase, wavelength, fluxerr)

    def phases(self):
        return self._phases.copy()

    def wavelengths(self):
        return self._wavelengths.copy()

    def flux(self, phase, wavelengths=None):
        """The model flux spectrum for the given parameters."""
        return self._model.y(phase, x1=wavelengths)

    def fluxerr(self, phase, wavelengths=None):
        """The flux error spectrum for the given parameters.
        """
        return self._modelerr.y(phase, x1=wavelengths)


class SALT2(Transient):
    """
    The SALT2 Type Ia supernova spectral timeseries model.

    Parameters
    ----------
    phases : numpy.ndarray
    wavelengths : numpy.ndarray
    M0, M1 : numpy.ndarray
        First and second model components.
        Shape must be (len(phases), len(wavelengths)).
    V00, V11, V01 : numpy.ndarray
        Variance of first component, second component and covariance.
        Shape must be (len(phases), len(wavelengths)).
    errorscale : numpy.ndarray, optional
        

"""

    def __init__(self, phases, wavelengths, M0, M1, V00, V11, V01,
                 errorscale=None):

        self._phases = phases
        self._wavelengths = wavelengths
        self._M0 = GridData(phases, wavelengths, M0)
        self._M1 = GridData(phases, wavelengths, M1)
        self._V00 = GridData(phases, wavelengths, V00)
        self._V11 = GridData(phases, wavelengths, V11)
        self._V01 = GridData(phases, wavelengths, V01)
        if errorscale is not None:
            self._errorscale = GridData(phases, wavelengths, errorscale)
        
        self._phases = phases
        self._wavelengths = wavelengths

    def phases(self):
        return self._phases.copy()

    def wavelengths(self):
        return self._wavelengths.copy()

    def flux(self, phase, wavelengths=None, x0=1.0, x1=0.0, c=0.0):
        """The model flux spectrum for the given parameters."""

        f0 = self._M0.y(phase, x1=wavelengths)
        f1 = self._M1.y(phase, x1=wavelengths)
        flux = x0 * (f0 + x1 * f1)
        flux *= self._extinction(wavelengths, c)

        return flux

    def fluxerr(self, phase, wavelengths=None, x0=1.0, x1=0.0, c=0.0):
        """The flux error spectrum for the given parameters.

        """
        v00 = self._V00.y(phase, x1=wavelengths)
        v11 = self._V11.y(phase, x1=wavelengths)
        v01 = self._V01.y(phase, x1=wavelengths)
        sigma = x0 * np.sqrt(v00 + x1*x1*v11 + 2*x1*v01)
        sigma *= self._extinction(wavelengths, c)
        ### sigma *= 1e-12   #- Magic constant from SALT2 code
        
        if self._errorscale is not None:
            sigma *= self._errorscale.y(phase, x1=wavelengths)
        
        # Hack adjustment to error (from SALT2 code)
        if phase < -7 or phase > 12:
            xx = np.nonzero(wavelengths < 3400)
            sigma[xx] *= 1000
        
        return sigma


    def _extinction(self, wavelengths=None, c=0.1,
                    params=(-0.336646, +0.0484495)):
        """
        From SALT2 code comments:
          ext = exp( color * constant * 
                    ( l + params(0)*l^2 + params(1)*l^3 + ... ) /
                    ( 1 + params(0) + params(1) + ... ) )
              = exp( color * constant *  numerator / denominator )
              = exp( color * expo_term ) 
        """

        const = math.log(10)/2.5

        if wavelengths is None: wavelengths = self._wavelengths
        wavelengths = np.array(wavelengths)

        wB = 4302.57
        wV = 5428.55
        wr = (wavelengths - wB) / (wV - wB)

        numerator = 1.0 * wr
        denominator = 1.0

        wi = wr * wr
        for p in params:
            numerator += wi * p
            denominator += p
            wi *= wr

        return np.exp(c * const * numerator / denominator)
