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
    modeldir : str
        Path to directory containing model files.
    """

    def __init__(self, modeldir):
        self._model = {}
        self._model['M0'] = GridData(modeldir + '/salt2_template_0.dat')
        self._model['M1'] = GridData(modeldir + '/salt2_template_1.dat')
        self._model['V00'] = GridData(modeldir + '/salt2_spec_variance_0.dat')
        self._model['V11'] = GridData(modeldir + '/salt2_spec_variance_1.dat')
        self._model['V01'] = GridData(modeldir +
                                      '/salt2_spec_covariance_01.dat')

        errorscalefile = modeldir + '/salt2_spec_dispersion_scaling.dat'
        if os.path.exists(errorscalefile):
            self._model['errorscale'] = GridData(errorscalefile)
        
        self._phases = self._model['M0'].x0()
        self._wavelengths = self._model['M0'].x1()

    def phases(self):
        return self._phases.copy()

    def wavelengths(self):
        return self._wavelengths.copy()

    def flux(self, phase, wavelengths=None, x0=1.0, x1=0.0, c=0.0):
        """The model flux spectrum for the given parameters."""

        f0 = self._model['M0'].y(phase, x1=wavelengths)
        f1 = self._model['M1'].y(phase, x1=wavelengths)
        flux = x0 * (f0 + x1 * f1)
        flux *= self._extinction(wavelengths, c)

        return flux

    def fluxerr(self, phase, wavelengths=None, x0=1.0, x1=0.0, c=0.0):
        """The flux error spectrum for the given parameters.

        """
        v00 = self._model['V00'].y(phase, x1=wavelengths)
        v11 = self._model['V11'].y(phase, x1=wavelengths)
        v01 = self._model['V01'].y(phase, x1=wavelengths)
        sigma = x0 * np.sqrt(v00 + x1*x1*v11 + 2*x1*v01)
        sigma *= self._extinction(wavelengths, c)
        ### sigma *= 1e-12   #- Magic constant from SALT2 code
        
        if 'errorscale' in self._model:
            sigma *= self._model['errorscale'].y(phase, x1=wavelengths)
        
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
