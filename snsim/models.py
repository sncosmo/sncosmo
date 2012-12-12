# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Classes to represent spectral models of astronomical transients."""

import abc
import os
import math

import numpy as np

from . import io
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
    m0file, m1file, v00file, v11file, v01file : str
        Path to files containing model components. Each file should have
        the format ``phase wavelength value`` on each line.
    errscalefile : str, optional
        Path to error scale file, same format as model component files.
        The default is ``None``, which means that the error scale will
        not be applied in the ``fluxerr()`` method.

    Notes
    -----
    The phase and wavelength values of the various components don't necessarily
    need to match. (In the most recent salt2 model data, they do not all 
    match.) The phase and wavelength values of the first model component
    (in ``m0file``) are taken as the "native" sampling of the model, even
    though these values might require interpolation of the other model
    components.
    """

    def __init__(self, m0file, m1file, v00file, v11file, v01file, 
                 errscalefile=None):

        self._model = {}

        components = ['M0', 'M1', 'V00', 'V11', 'V01', 'errscale']
        filenames = [m0file, m1file, v00file, v11file, v01file, errscalefile]
        for component, filename in zip(components, filenames):

            # If the filename is None, that component is left out of the model
            if filename is None: continue

            # Get the model component from the file
            phases, wavelengths, values = io.read_griddata_txt(filename)
            self._model[component] = GridData(phases, wavelengths, values)

            # The "native" phases and wavelengths of the model are those
            # of the first model component.
            if component == 'M0':
                self._phases = phases
                self._wavelengths = wavelengths    


    def phases(self, copy=False):
        """Return native phases of the model in days."""
        if copy: return self._phases.copy()
        else: return self._phases


    def wavelengths(self, copy=False):
        """Return native wavelengths of the model in Angstroms."""
        if copy: return self._wavelengths.copy()
        else: return self._wavelengths

        
    def flux(self, phase, wavelengths=None, x0=1.0, x1=0.0, c=0.0):
        """The model flux spectrum for the given parameters."""

        if wavelengths is None:
            wavelengths = self._wavelengths
        f0 = self._model['M0'].y(phase, x1=wavelengths)
        f1 = self._model['M1'].y(phase, x1=wavelengths)
        flux = x0 * (f0 + x1 * f1)
        flux *= self._extinction(wavelengths, c)

        return flux

    def fluxerr(self, phase, wavelengths=None, x0=1.0, x1=0.0, c=0.0):
        """The flux error spectrum for the given parameters.
        """
        if wavelengths is None:
            wavelengths = self._wavelengths

        v00 = self._model['V00'].y(phase, x1=wavelengths)
        v11 = self._model['V11'].y(phase, x1=wavelengths)
        v01 = self._model['V01'].y(phase, x1=wavelengths)
        sigma = x0 * np.sqrt(v00 + x1*x1*v11 + 2*x1*v01)
        sigma *= self._extinction(wavelengths, c)
        ### sigma *= 1e-12   #- Magic constant from SALT2 code
        
        if 'errscale' in self._model:
            sigma *= self._model['errscale'].y(phase, x1=wavelengths)
        
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

        if wavelengths is None:
            wavelengths = self._wavelengths
        wavelengths = np.asarray(wavelengths)

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
