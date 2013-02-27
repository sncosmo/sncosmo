# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Classes to represent spectral models of astronomical transients."""

import abc
import os
import math

import numpy as np
from astropy.utils.misc import isiterable

from .utils import GridData, read_griddata
from .spectral import Spectrum
from . import registry

__all__ = ['get_model', 'Model', 'TimeSeriesModel', 'SALT2Model']

def get_model(name, version=None):
    """Retrieve a model from the registry by name.

    Parameters
    ----------
    name : str
        Name of model in the registry.
    version : str, optional
        Version identifier for models with multiple versions. Default is
        `None` which corresponds to the latest, or only, version.
    """
    return registry.retrieve(Model, name, version=version)


class Model(object):
    """An abstract base class for transient models.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        self._params = dict(z=0., fluxscaling=1.)
        self._refphase = 0.
        self._cosmo = cosmology.get_current()

    @abc.abstractmethod
    def set(self, **params):
        """Set the parameters of the model.

        `z`, `absmag`, `fluxscaling` are common to all models."""
        pass

    @abc.abstractmethod
    def _model_fluxdensity(self, phase=None, dispersion=None, extend=False):
        """Return the model flux density (without any scaling or redshifting).
        """
        pass

    @abc.abstractmethod
    def _model_fluxdensityerror(self, phase=None, dispersion=None,
                                extend=False):
        """Return the model flux density error (without any scaling or
        redshifting. Return `None` if the error is undefined.
        """
        pass

    @property
    def cosmo(self):
        """Cosmology used to calculate luminosity distance."""
        return self._cosmo

    @cosmo.setter
    def cosmo(self, new_cosmology):
        self._cosmo = new_cosmology

    @property
    def refphase(self):
        """Phase at which absolute magnitude is evaluated.

        Typically set to the phase of maximum light in some band."""
        return self._refphase

    @refphase.setter
    def refphase(self, new_refphase):
        self._refphase = new_refphase

    def phase(self, restframe=False):
        """Return native phases of the model."""
        if restframe:
            return self._phase
        else:
            return self._phase * (1. + self._params['z'])

    #TODO assumes wavelength
    def dispersion(self, restframe=False):
        """Return native dispersion grid of the model."""
        if restframe:
            return self._dispersion
        else:
            return self._dispersion * (1. + self._params['z'])

    def fluxdensity(self, phase=None, dispersion=None, extend=False,
                    restframe=False):
        """The model flux spectral density."""

        if restframe or self.params['z'] == 0.:
            return (self._params['fluxscaling'] *
                    self._model_fluxdensity(phase, dispersion, extend=extend))
        if self._params['z'] is None:
            raise ValueError('observer frame requested, but redshift '
                             'undefined.')
        if phase is not None:
            phase /= (1. + self._params['z'])
        if dispersion is not None:
            dispersion /= (1. + self._params['z'])

        f = self._model_fluxdensity(phase, dispersion, extend=extend)

        # Apply rest-frame flux scaling, then redshift
        dist = self._cosmo.luminosity_distance(self._params['z'])
        f *= (self._params['fluxscaling'] *
              (1.e-5 / dist) ** 2 / (1. + self._params['z']))

    def fluxdensityerror(self, phase=None, dispersion=None, extend=False,
                         restframe=False):
        """The error on the model flux spectral density."""

        if restframe or self.params['z'] == 0.:
            f = self._model_fluxdensityerror(phase, dispersion, extend=extend)
            if f is None: return f
            return self._params['fluxscaling'] * f

        if self._params['z'] is None:
            raise ValueError('observer frame requested, but redshift '
                             'undefined.')

        if phase is not None:
            phase /= (1. + self._params['z'])
        if dispersion is not None:
            dispersion /= (1. + self._params['z'])

        f = self._model_fluxdensityerror(phase, dispersion, extend=extend)
        if f is None: return f

        # Apply rest-frame flux scaling, then redshift
        dist = self._cosmo.luminosity_distance(self._params['z'])
        f *= (self._params['fluxscaling'] *
              (1.e-5 / dist) ** 2 / (1. + self._params['z']))

    def spectrum(self, phase, dispersion=None, restframe=False, 
                 include_error=False):
        """Return a `Spectrum` generated from the model at the given phase.

        This is equivalent to

            >>> Spectrum(m.dispersion(), m.fluxdensity(phase))

        """
        d = self.dispersion(restframe=restframe)
        f = self.fluxdensity(phase, dispersion=dispersion, restframe=restframe)
        if include_error:
            fe = self.fluxdensityerror(phase, dispersion=dispersion,
                                       restframe=restframe)
        return Spectrum(d, f, error=fe, z=self._params['z'])

    def flux(self, phase, band):
        """Flux (ph/cm^2/s) at the given phase(es) through the given 
        bandpass(es)"""

        phase, band = np.broadcast_arrays(phase, band)
        ndim = phase.ndim
        phase = phase.ravel()
        band = band.ravel()

        flux_values = np.empty(phase.shape, dtype=np.float)

        for i, ph, b in zip(range(len(phase)), phase, band):
            s = spectrum(ph)
            flux_values[i] = s.flux(b)
            
        if ndim == 0:
            return flux_values[0]
        return flux_values
            
        

    def magnitude(self, phase, band, magsys):
        """Magnitudes at the given phase(es) through the given 
        bandpass(es), and for the given magnitude systems."""
        pass


    def __call__(self, **params):
        """Return a shallow copy of the model with parameters set.

            >>> m2 = m(**params)
        
        is equivalent to:

            >>> m2 = copy.copy(m)
            >>> m2.set(**params)

        """
        m = copy.copy(self)
        m.set(**params)
        return m

    def __repr__(self):
        return "<{}>".format(self.__class__.__name__)

    def __str__(self):
        result = """
        Model class: {}
        Model name: {}
        Model version: {}
        Restframe phases: {:.6g}..{:.6g} ({:d} points)
        Restframe dipsersion: {:.6g}..{:.6g} ({:d} points) 
        Reference phase: {}
        Cosmology: {}
        Parameters: {}
        """.format(self.__class__.__name__,
                   self._name, self._version,
                   self._phase[0], self._phase[-1], len(self._phase),
                   self._dispersion[0], self._dispersion[-1],
                   len(self._dispersion),
                   self._refphase,
                   self._cosmo,
                   self._params)
        return result


class TimeSeriesModel(Model):
    """A single-component spectral time series model.

    Parameters
    ----------
    phase : `~numpy.ndarray`
        Phases in days.
    dispersion : `~numpy.ndarray`
        Wavelengths in Angstroms.
    fluxdensity : `~numpy.ndarray`
        Model spectral flux density in (erg/s/cm^2/Angstrom).
        Must have shape `(num_phases, num_dispersion)`.
    fluxdensityerror : `~numpy.ndarray`, optional
        Model error on `fluxdensity`. Must have same shape as `fluxdensity`.
        Default is `None`.
    """

    def __init__(self, phase, dispersion, fluxdensity,
                 fluxdensityerror=None, name=None, version=None):
        super()
        self._name = name
        self._version = version
        self._phase = phase
        self._dispersion = dispersion
        self._model = GridData(phase, dispersion, fluxdensity)

        if fluxdensityerror is None:
            self._modelerror = None
        else:
            self._modelerror = GridData(phase, dispersion, fluxdensityerror)

    def _model_fluxdensity(self, phase=None, dispersion=None, extend=False):
        """Return the model flux density (without any scaling or redshifting).
        """
        return self._model(phase, dispersion, extend=extend)

    def _model_fluxdensityerror(self, phase=None, dispersion=None, extend=True):
        """Return the model flux density error (without any scaling or
        redshifting. Return `None` if the error is undefined.
        """
        if self._modelerror is None:
            return None
        return self._modelerror(phase, dispersion, extend=extend)


class SALT2Model(Model):
    """
    The SALT2 Type Ia supernova spectral timeseries model.

    Parameters
    ----------
    modeldir : str
        Path to files containing model components. Each file should have
        the format:
 
            phase wavelength value

        on each line. If you want to give the absolute path to each file,
        set to `None`.
    m0file, m1file, v00file, v11file, v01file : str or fileobj, optional
        Filenames of various model components. Defaults are:

        * m0file = 'salt2_template_0.dat'
        * m1file = 'salt2_template_1.dat'
        * v00file = 'salt2_spec_variance_0.dat'
        * v11file = 'salt2_spec_variance_1.dat'
        * v01file = 'salt2_spec_covariance_01.dat'

    errscalefile : str, optional
        Name of error scale file, same format as model component files.
        The default is ``None``, which means that the error scale will
        not be applied in the ``fluxerr()`` method. This is only used for
        template versions 1.1 and 1.0, not 2.0+.

    Notes
    -----
    The phase and wavelength values of the various components don't necessarily
    need to match. (In the most recent salt2 model data, they do not all 
    match.) The phase and wavelength values of the first model component
    (in ``m0file``) are taken as the "native" sampling of the model, even
    though these values might require interpolation of the other model
    components.
    """

    def __init__(self, modeldir=None,
                 m0file='salt2_template_0.dat',
                 m1file='salt2_template_1.dat',
                 v00file='salt2_spec_variance_0.dat',
                 v11file='salt2_spec_variance_1.dat',
                 v01file='salt2_spec_covariance_01.dat',
                 errscalefile=None, name=None, version=None):
        super()
        self._params['c'] = 0.
        self._params['x1'] = 0.
        self._name = name
        self._version = version
        self._model = {}

        components = ['M0', 'M1', 'V00', 'V11', 'V01', 'errscale']
        names_or_objs = [m0file, m1file, v00file, v11file, v01file,
                         errscalefile]

        # Make filenames into full paths.
        if modeldir is not None:
            for i in range(len(names_or_objs)):
                if (names_or_objs[i] is not None and
                    isinstance(names_or_objs[i], basestring)):
                    names_or_objs[i] = os.path.join(modeldir, names_or_objs[i])

        for component, name_or_obj in zip(components, names_or_objs):

            # If the filename is None, that component is left out of the model
            if name_or_obj is None: continue

            # Get the model component from the file
            phase, wavelength, values = read_griddata(name_or_obj)
            self._model[component] = GridData(phase, wavelength, values)

            # The "native" phases and wavelengths of the model are those
            # of the first model component.
            if component == 'M0':
                self._phase = phase
                self._dispersion = wavelength

    def _model_fluxdensity(self, phase=None, dispersion=None, extend=False):
        """Return the model flux density (without any scaling or redshifting).
        """

        if phase is None:
            phase = self._phase
        if dispersion is None:
            dispersion = self._dispersion
        f0 = self._model['M0'](phase, dispersion, extend)
        f1 = self._model['M1'](phase, dispersion, extend)
        flux = f0 + self._params['x1'] * f1
        flux *= self._extinction(dispersion, self._params['c'])
        return flux

    def _model_fluxdensityerror(self, phase=None, dispersion=None, extend=True):
        """Return the model flux density error (without any scaling or
        redshifting. Return `None` if the error is undefined.
        """
        if phase is None:
            phase = self._phase
        if dispersion is None:
            dispersion = self._dispersion
        phase = np.asarray(phase)
        dispersion = np.asarray(dispersion)

        v00 = self._model['V00'](phase, dispersion)
        v11 = self._model['V11'](phase, dispersion)
        v01 = self._model['V01'](phase, dispersion)

        #TODO apply x0 correctly
        # used to be sigma = x0 * np.sqrt(v00 + ...)
        x1 = self._params['x1']
        sigma = np.sqrt(v00 + x1 * x1 * v11 + 2 * x1 * v01)
        sigma *= self._extinction(dispersion, self._params['c'])
        ### sigma *= 1e-12   #- Magic constant from SALT2 code
        
        if 'errscale' in self._model:
            sigma *= self._model['errscale'](phase, dispersion)
        
        # Hack adjustment to error (from SALT2 code)
        #TODO: figure out a way to do this.

        #if phase < -7. or phase > 12.:
        #    idx = (dispersion < 3400.)
        #    sigma[idx] *= 1000.
        
        return sigma


    def _extinction(self, wavelengths, c,
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
