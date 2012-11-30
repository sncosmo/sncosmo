# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Classes to represent spectral models of astronomical transients."""

import abc
import os
import glob
import numpy as np

__all__ = ['SALT2Model']

# Not yet sure the best way to implement the model() function to return
# a specific model based on strings, so for now I'm commenting this out:

#def model(modelname, modelpath=None):
#    """Create and return a Model class instance."""
#
#    _builtin_models = {'salt2': SALT2Model}
#
#    if modelname in _builtin_models:
#        Model = _builtin_models[modelname]
#        return Model(modelpath)
#    else:
#        raise ValueError('model name "{}" is not a built-in model.')


class TransientModel(object):
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


class SALT2Model(TransientModel):
    """
    An interface class to the SALT2 supernovae spectral timeseries model.
    """

    def __init__(self, modeldir=None):
        """
        Create a SALT2 model, using the model files found under modeldir,
        which is equivalent to the PATHMODEL environment variable for SALT2.
        i.e. it expects to find files in modeldir/salt2*/salt2*.dat
        
        It will use $PATHMODEL/salt2 if it is there, otherwise it will
        use the latest version in $PATHMODEL/salt2* .
        """
        if modeldir is None:
            if 'PATHMODEL' not in os.environ:
                raise ValueError("If you don't specify modeldir, you must "
                                 "have salt2 PATHMODEL env var defined")
            pathmodel = os.environ['PATHMODEL']
            if os.path.exists(pathmodel + '/salt2/'):
                modeldir =  pathmodel + '/salt2/'
            else:
                models = sorted(glob.glob(pathmodel + '/salt2*'))
                if len(models) == 0:
                    raise ValueError("Model directory %s/salt2* not found" %
                                     pathmodel)
                modeldir = models[-1]

            print "Using model", modeldir

        self._model = {}
        self._model['M0'] = TimeSeries(modeldir + '/salt2_template_0.dat')
        self._model['M1'] = TimeSeries(modeldir + '/salt2_template_1.dat')
        self._model['V00'] = TimeSeries(modeldir + '/salt2_spec_variance_0.dat')
        self._model['V11'] = TimeSeries(modeldir + '/salt2_spec_variance_1.dat')
        self._model['V01'] = TimeSeries(modeldir +
                                        '/salt2_spec_covariance_01.dat')
        errorscale = modeldir + '/salt2_spec_dispersion_scaling.dat'
        if os.path.exists(errorscale):
            self._model['errorscale'] = TimeSeries(errorscale)
        
        self._wavelengths = self._model['M0'].wavelengths()
        

    def wavelengths(self):
        return self._wavelengths.copy()

    def flux(self, phase, wavelengths=None, x0=1.0, x1=0.0, c=0.0):
        """Return the model (wavelength, flux) spectrum for these parameters"""
        if wavelengths is None:
            wavelengths = self._wavelengths
        f0 = self._model['M0'].spectrum(phase, wavelengths=wavelengths)
        f1 = self._model['M1'].spectrum(phase, wavelengths=wavelengths)
        flux = x0 * (f0 + x1 * f1)

        flux *= self._extinction(wavelengths, c)

        return flux

    def error(self, phase, wavelengths=None, x0=1.0, x1=0.0, c=0.0):
        """
        Return flux error spectrum for given parameters
        """
        if wavelengths is None:
            wavelengths = self._wavelengths
            
        v00 = self._model['V00'].spectrum(phase, wavelengths=wavelengths)
        v11 = self._model['V11'].spectrum(phase, wavelengths=wavelengths)
        v01 = self._model['V01'].spectrum(phase, wavelengths=wavelengths)

        sigma = x0 * np.sqrt(v00 + x1*x1*v11 + 2*x1*v01)
        sigma *= self._extinction(wavelengths, c)
        ### sigma *= 1e-12   #- Magic constant from SALT2 code
        
        if 'errorscale' in self._model:
            sigma *= self._model['errorscale'].spectrum(phase,
                                                        wavelengths=wavelengths)
        
        # Hack adjustment to error (from SALT2 code)
        if phase < -7 or phase > 12:
            xx = np.nonzero(wavelengths < 3400)
            sigma[xx] *= 1000
        
        return sigma

    def variance(self, phase, wavelengths=None, x0=1.0, x1=0.0, c=0.0):
        """Return model flux variance for given parameters.
        """
        return self.error(phase, wavelengths=wavelengths, x0=x0, x1=x1, c=c)**2


    def _extinction(self, w, c=0.1, params=(-0.336646, +0.0484495)):
        """
        From SALT2 code comments:
          ext = exp( color * constant * 
                    ( l + params(0)*l^2 + params(1)*l^3 + ... ) /
                    ( 1 + params(0) + params(1) + ... ) )
              = exp( color * constant *  numerator / denominator )
              = exp( color * expo_term ) 
        """
        from math import log
        const = log(10)/2.5

        if type(w) != np.ndarray:
            w = np.array(w)

        wB = 4302.57
        wV = 5428.55
        wr = (w - wB) / (wV - wB)

        numerator = 1.0 * wr
        denominator = 1.0

        wi = wr * wr
        for p in params:
            numerator += wi * p
            denominator += p
            wi *= wr

        return np.exp(c * const * numerator / denominator)


class TimeSeries(object):
    """A series of values associated with a phase and a wavelength,
    e.g. a time series of spectra or a time series of their errors
    
    Parameters
    ----------
    filename : str
        ASCII file with grid data in the form `phase wavelength value`
    """
    
    def __init__(self, filename):
        self._wavelenghts = None  # Wavelenghts of first day, assume others
                                  # are the same.
        self._phases = []  # Phases in the model file
        self._spectra = []  # One spectrum interpolation function for
                            # each phase

        currentday = None
        w = []
        flux = []
        for line in open(filename):
            day, wavelength, value = map(float, line.split())
            if currentday == None:
                currentday = day

            if day != currentday:
                self._phases.append(currentday)
                self._spectra.append(lambda x: np.interp(x, w, flux))
                if self._wavelenghts is None:
                    self._wavelengths = np.array(w)
                currentday = day
                w = []
                flux = []
            
            w.append(wavelength)
            flux.append(value)
            
        # Get the last day of information in there 
        self._phases.append(currentday)
        self._spectra.append(lambda x: np.interp(x, w, flux))


    def spectrum(self, phase, wavelengths=None, extend=True):
        """
        Return spectrum at requested phase and wavelengths.
        Raise ValueError if phase is out of range of model unless extend=True.
        """

        # Bounds check first
        if phase < self._phases[0] and not extend:
            raise ValueError("phase %.2f before first model phase %.2f" %
                             (phase, self._phases[0]))
        if phase > self._phases[-1] and not extend:
            raise ValueError("phase %.2f after last model phase %.2f" %
                             (phase, self._phases[-1]))

        # Use default wavelengths if none are specified
        if wavelengths is None:
            wavelengths = self._wavelengths

        # Check if requested phase is out of bounds or exactly in the list
        if phase in self._phases:
            iphase = self._phases.index(phase)
            return self._spectra[iphase](wavelengths)
        elif phase < self._phases[0]:
            return self._spectra[0](wavelengths)
        elif phase > self._phases[-1]:
            return self._spectra[-1](wavelengths)
            
        # If we got this far, we need to interpolate phases
        i = np.searchsorted(self._phases, phase)
        speclate = self._spectra[i](wavelengths)
        specearly = self._spectra[i - 1](wavelengths)
        dphase = ((phase - self._phases[i - 1]) /
                  (self._phases[i] - self._phases[i - 1]))
        dspec = speclate - specearly
        spec = specearly + dphase * dspec
        
        return spec

    def wavelengths(self):
        """Return array of wavelengths sampled in the model"""
        return self._wavelengths.copy()

    def phases(self):
        """Return array of phases sampled in the model"""
        return np.array(self._phases)
        
    def grid(self):
        """Return a 2D array of spectrum[phase, wavelength]"""
        nspec = len(self._phases)
        nwave = len(self._wavelengths)
        z = np.zeros((nspec, nwave), dtype=np.float64)
        for i, spec in enumerate(self._spectra):
            z[i, :] = spec(self._wavelengths)
        return z



