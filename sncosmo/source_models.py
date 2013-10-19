# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Classes to represent spectral models of astronomical transients."""

import abc
import os
import copy

import numpy as np
from scipy.interpolate import (InterpolatedUnivariateSpline as Spline1d,
                               RectBivariateSpline as Spline2d)

from .io import read_griddata

__all__ = ['SourceModel', 'TimeSeriesModel', 'StretchModel',
           'SALT2Model']

class SourceModel(object):
    """An abstract base class for transient models.
    
    A "transient model" in this case is the spectral time evolution
    of a source as a function of an arbitrary number of parameters.

    This is an abstract base class -- You can't create instances of
    this class. Instead, you must work with subclasses such as
    `TimeSeriesModel`. Subclasses must define (at minimum): 

    * `__init__()`
    * `_flux()` or `flux()`
    * `_param_names` (list of str)
    * `_parameters` (`numpy.ndarray`)
    * `_phase` (`numpy.ndarray`)
    * `_wave` (`numpy.ndarray`)
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def param_names(self):
        return self._param_names

    @property
    def parameters(self):
        return self._parameters[:]

    @property
    def phases(self):
        return self._phase

    @property
    def wavelengths(self):
        return self._wave

    def flux(self, phase=None, wave=None):
        """The spectral flux density at the given phase and wavelength values.

        Parameters
        ----------
        phase : float or list_like, optional
            Phase(s) in days. If `None` (default), the native phases of
            the model are used.
        wave : float or list_like, optional
            Wavelength(s) in Angstroms. If `None` (default), the native
            wavelengths of the model are used.

        Returns
        -------
        flux : float or `~numpy.ndarray`
            Spectral flux density values in ergs / s / cm^2 / Angstrom.
        """
        if phase is None:
            phase = self._phase
        if wave is None:
            wave = self._wave
        flux = self._flux(phase, wave)

        # Check dimensions of phase, wave for return value
        # (1, 1) -> ndim=2
        # (1, 0) -> ndim=2
        # (0, 1) -> ndim=1
        # (0, 0) -> float
        if phase.ndim == 0:
            if wave.ndim == 0:
                return flux[0, 0]
            return flux[0, :]
        return flux

    def __copy__(self):
        copied_model = copy.copy(self)
        copied_model._param_names = self._param_names[:]
        copied_model._parameters = self._parameters.copy()
        return copied_model

class TimeSeriesModel(SourceModel):
    """A single-component spectral time series model.

    Parameters
    ----------
    phase : `~numpy.ndarray`
        Phases in days.
    wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
    flux : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape ``(num_phases, num_wave)``.
    name : str, optional
        Name of the model. Default is `None`.
    version : str, optional
        Version of the model. Default is `None`.
    """

    _param_names = ['fscale']
    _parameters = np.array([1.])

    def __init__(self, phase, wave, flux,
                 name=None, version=None):

        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._model_flux = Spline2d(phase, wave, flux, kx=2, ky=2)

    def _flux(self, phase, wave):
        return self._parameters[0] * self._model_flux(phase, wave)

class StretchModel(SourceModel):
    """A single-component spectral time series model, that "stretches".

    Parameters
    ----------
    phase : `~numpy.ndarray`
        Phases in days.
    wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
    flux : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape `(num_phases, num_disp)`.
    """

    _param_names = ['fscale', 's']
    _parameters = np.array([1., 1.])

    def __init__(self, phase, wave, flux, name=None, version=None):
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._model_flux = Spline2d(phase, wave, flux, kx=2, ky=2)

    @property
    def phases(self):
        return self._parameters[1] * self._phases

    # Need to override this to deal with phases.
    def flux(self, phase=None, wave=None):
        if phase is None:
            phase = self._phase
        else:
            phase = phase / self._parameters[1]
        if wave is None:
            wave = self._wave
        flux  = self._parameters[0] * self._model_flux(phase, wave)

        # Check dimensions of phase, disp for return value
        # (1, 1) -> ndim=2
        # (1, 0) -> ndim=2
        # (0, 1) -> ndim=1
        # (0, 0) -> float
        if phase.ndim == 0:
            if disp.ndim == 0:
                return flux[0, 0]
            return flux[0, :]
        return flux

class SALT2Model(SourceModel):
    """The SALT2 Type Ia supernova spectral timeseries model.

    Parameters
    ----------
    modeldir : str, optional
        Directory path containing model component files. Default is `None`,
        which means that no directory is prepended to filenames when
        determining their path.
    m0file, m1file, v00file, v11file, v01file, clfile : str or fileobj, optional
        Filenames of various model components. Defaults are:

        * m0file = 'salt2_template_0.dat'
        * m1file = 'salt2_template_1.dat'
        * v00file = 'salt2_spec_variance_0.dat'
        * v11file = 'salt2_spec_variance_1.dat'
        * v01file = 'salt2_spec_covariance_01.dat'
        * clfile = 'salt2_color_correction.dat'

        The first five files should have the format
        ``<phase> <wavelength> <value>`` on each line. The colorlaw file
        (clfile) has a different format.
    errscalefile : str, optional
        Name of error scale file, same format as model component files.
        The default is ``None``, which means that the error scale will
        not be applied in the ``fluxerr()`` method. This is only used for
        template versions 1.1 and 1.0, not 2.0+.

    Notes
    -----
    The phase and wavelength values of the various components don't
    necessarily need to match. (In the most recent salt2 model data,
    they do not all match.) The phase and wavelength values of the
    first model component (in ``m0file``) are taken as the "native"
    sampling of the model, even though these values might require
    interpolation of the other model components.
    """

    _param_names = ['fscale', 'x1', 'c']
    _parameters = np.array([1., 0., 0.])

    def __init__(self, modeldir=None,
                 m0file='salt2_template_0.dat',
                 m1file='salt2_template_1.dat',
                 v00file='salt2_spec_variance_0.dat',
                 v11file='salt2_spec_variance_1.dat',
                 v01file='salt2_spec_covariance_01.dat',
                 clfile='salt2_color_correction.dat',
                 errscalefile=None, name=None, version=None):
        self.name = name
        self.version = version
        self._model = {}
        components = ['M0', 'M1', 'V00', 'V11', 'V01', 'errscale', 'clfile']
        names_or_objs = [m0file, m1file, v00file, v11file, v01file,
                         errscalefile, clfile]

        # Make filenames into full paths.
        if modeldir is not None:
            for i in range(len(names_or_objs)):
                if (names_or_objs[i] is not None and
                    isinstance(names_or_objs[i], basestring)):
                    names_or_objs[i] = os.path.join(modeldir, names_or_objs[i])

        # Read components gridded in (phase, wavelength)
        for component, name_or_obj in zip(components[:-1], names_or_objs[:-1]):

            # If the filename is None, that component is left out of the model
            if name_or_obj is None: continue

            # Get the model component from the file
            phase, wave, values = read_griddata(name_or_obj)
            self._model[component] = Spline2d(phase, wave, values, kx=2, ky=2)

            # The "native" phases and wavelengths of the model are those
            # of the first model component.
            if component == 'M0':
                self._phase = phase
                self._wave = wave
            
        # Set the colorlaw based on the "color correction" file.
        self._set_colorlaw_from_file(names_or_objs[-1])

        # add extinction component
        cl = self._colorlaw(self._wave)
        clbase = 10. ** (-0.4 * cl)
        self._model['clbase'] = Spline1d(self._wave, dust_trans_base, k=2)

    def _flux(self, phase, wave):
        m0 = self._model['M0'](phase, wave)
        m1 = self._model['M1'](phase, wave)
        return (self._parameters[0] *
                (m0 + self._parameters[1] * m1) *
                self._model['clbase'](wave) ** self._parameters[2])

    def _set_colorlaw_from_file(self, name_or_obj):
        """Read color law file and set the internal colorlaw function,
        as well as some parameters used in that function.

        self._colorlaw (function)
        self._B_WAVELENGTH (float)
        self._V_WAVELENGTH (float)
        self._colorlaw_coeffs (list of float)
        self._colorlaw_range (tuple) [default is (3000., 7000.)]
        """

        self._B_WAVELENGTH = 4302.57
        self._V_WAVELENGTH = 5428.55

        # Read file
        if isinstance(name_or_obj, basestring):
            f = open(name_or_obj, 'rb')
        else:
            f = name_or_obj
        words = f.read().split()
        f.close()

        # Get colorlaw coeffecients.
        npoly = int(words[0])
        self._colorlaw_coeffs = [float(word) for word in words[1: 1 + npoly]]
    
        # Look for keywords in the rest of the file.
        version = 0
        colorlaw_range = [3000., 7000.]
        for i in range(1+npoly, len(words)):
            if words[i] == 'Salt2ExtinctionLaw.version':
                version = int(words[i+1])
            if words[i] == 'Salt2ExtinctionLaw.min_lambda':
                colorlaw_range[0] = float(words[i+1])
            if words[i] == 'Salt2ExtinctionLaw.max_lambda':
                colorlaw_range[1] = float(words[i+1])

        # Set extinction function to use.
        if version == 0:
            self._colorlaw = self._colorlaw_v0
        elif version == 1:
            self._colorlaw = self._colorlaw_v1
            self._colorlaw_range = colorlaw_range
        else:
            raise Exception('unrecognized Salt2ExtinctionLaw.version: {}'
                            .format(version))


    def _colorlaw_v0(self, wave):
        """Return the extinction in magnitudes as a function of wavelength,
        for c=1. This is the version 0 extinction law used in SALT2 1.0 and
        1.1 (SALT2-1-1).

        Notes
        -----
        From SALT2 code comments:

            ext = exp(color * constant * 
                      (l + params(0)*l^2 + params(1)*l^3 + ... ) /
                      (1 + params(0) + params(1) + ... ) )
                = exp(color * constant *  numerator / denominator )
                = exp(color * expo_term ) 
        """

        l = ((wave - self._B_WAVELENGTH) /
             (self._V_WAVELENGTH - self._B_WAVELENGTH))

        coeffs = [0., 1.]
        coeffs.extend(self._colorlaw_coeffs)
        coeffs = np.flipud(coeffs)
        numerator = np.polyval(coeffs, l)  # 0 + 1 * l + p[0] * l^2 + ...
        denominator = coeffs.sum()         # 0 + 1 + p[0] + p[1] + ...

        return -numerator / denominator

    def _colorlaw_v1(self, wave):
        """Return the  extinction in magnitudes as a function of wavelength,
        for c=1. This is the version 1 extinction law used in SALT2 2.0
        (SALT2-2-0).

        Notes
        -----
        From SALT2 code comments:

        if(l_B<=l<=l_R):
            ext = exp(color * constant *
                      (alpha*l + params(0)*l^2 + params(1)*l^3 + ... ))
                = exp(color * constant * P(l))

            where alpha = 1 - params(0) - params(1) - ...

        if (l > l_R):
            ext = exp(color * constant * (P(l_R) + P'(l_R) * (l-l_R)))
        if (l < l_B):
            ext = exp(color * constant * (P(l_B) + P'(l_B) * (l-l_B)))
        """

        v_minus_b = self._V_WAVELENGTH - self._B_WAVELENGTH

        l = (wave - self._B_WAVELENGTH) / v_minus_b
        l_lo = (self._colorlaw_range[0] - self._B_WAVELENGTH) / v_minus_b
        l_hi = (self._colorlaw_range[1] - self._B_WAVELENGTH) / v_minus_b

        alpha = 1. - sum(self._colorlaw_coeffs)
        coeffs = [0., alpha]
        coeffs.extend(self._colorlaw_coeffs)
        coeffs = np.array(coeffs)
        prime_coeffs = (np.arange(len(coeffs)) * coeffs)[1:]

        extinction = np.empty_like(wave)
        
        # Blue side
        idx_lo = l < l_lo
        p_lo = np.polyval(np.flipud(coeffs), l_lo)
        pprime_lo = np.polyval(np.flipud(prime_coeffs), l_lo)
        extinction[idx_lo] = p_lo + pprime_lo * (l[idx_lo] - l_lo)

        # Red side
        idx_hi = l > l_hi
        p_hi = np.polyval(np.flipud(coeffs), l_hi)
        pprime_hi = np.polyval(np.flipud(prime_coeffs), l_hi)
        extinction[idx_hi] = p_hi + pprime_hi * (l[idx_hi] - l_hi)
        
        # In between
        idx_between = np.invert(idx_lo | idx_hi)
        extinction[idx_between] = np.polyval(np.flipud(coeffs), l[idx_between])

        return -extinction

    def colorlaw(self, wave):
        """Return the value of the CL function for the given wavelengths.

        Parameters
        ----------
        wave : float or list_like

        Returns
        -------
        colorlaw : float or `~numpy.ndarray`
            Values of colorlaw function, which can be interpreted as extinction
            in magnitudes.
            
        Notes
        -----
        Note that this is the "exact" colorlaw. For performance reasons, when
        calculating the model flux, a spline fit to this function output is
        used rather than the function itself. Therefore this will not be
        *exactly* equivalent to the color law used when evaluating the model
        flux. (It will however be exactly equivalent at the native model
        wavelengths.)
        """

        wave = np.asarray(wave)
        if wave.ndim == 0:
            return self._colorlaw(np.ravel(wave))[0]
        else:
            return self._colorlaw(wave)
