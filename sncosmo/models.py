# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Classes to represent spectral models of astronomical transients."""

import abc
import os
from copy import copy as cp
from textwrap import dedent

import numpy as np
from scipy.interpolate import (InterpolatedUnivariateSpline as Spline1d,
                               RectBivariateSpline as Spline2d)
from astropy.utils import OrderedDict as odict
from astropy import (cosmology,
                     units as u,
                     constants as const)

from .io import read_griddata
from . import registry
from .spectral import get_bandpass, get_magsystem
from .extinction import extinction

__all__ = ['get_sourcemodel', 'get_obsmodel', 'get_model',
           'SourceModel', 'TimeSeriesModel', 'StretchModel', 'SALT2Model',
           'ObsModel', 'PropagationEffect', 'InterpolatedRvDust']

HC_ERG_AA = const.h.cgs.value * const.c.to(u.AA / u.s).value

def get_sourcemodel(name, version=None, copy=False):
    """Retrieve a model from the registry by name.

    Parameters
    ----------
    name : str
        Name of model in the registry.
    version : str, optional
        Version identifier for models with multiple versions. Default is
        `None` which corresponds to the latest, or only, version.
    copy : bool, optional
        If True and if `name` is already a Model instance, return a copy of
        it. (If `name` is a str a copy of the instance
        in the registry is always returned, regardless of the value of this
        parameter.) Default is False.
    """
    
    # If we need to retrieve from the registry, we want to return a shallow
    # copy, in order to keep the copy in the registry "pristene". However, we
    # *don't* want a shallow copy otherwise. Therefore,
    # we need to check if `name` is already an instance of Model before 
    # going to the registry, so we know whether or not to make a shallow copy.
    if isinstance(name, SourceModel):
        if copy:
            return cp(name)
        else:
            return name
    else:
        return cp(registry.retrieve(SourceModel, name, version=version))

# TODO maybe put ObsModels directly in the registry.
# TODO clean up 
def get_obsmodel(name, version=None, copy=False):
    """Build an ObsModel based on the named SourceModel."""

    if isinstance(name, ObsModel):
        if copy:
            return cp(name)
        else:
            return name

    source = cp(registry.retrieve(SourceModel, name, version=version))
    if source.__class__.__name__ == 'SALT2Model':
        effects = [InterpolatedRvDust()]
        effect_names = ['mw']
        effect_frames = ['obs']
    else:
        effects = [InterpolatedRvDust(), InterpolatedRvDust()]
        effect_names = ['host', 'mw']
        effect_frames = ['rest', 'obs']
    return ObsModel(source, effects=effects, effect_names=effect_names,
                    effect_frames = effect_frames)

# TODO: deprecate this
get_model = get_obsmodel


def _bandflux(model, band, time_or_phase, zp, zpsys):
    """Support function for bandflux in SourceModel and ObsModel.
    This is necessary to have outside because ``phase`` is used in SourceModel
    and ``time`` is used in ObsModel.
    """

    if zp is not None and zpsys is None:
        raise ValueError('zpsys must be given if zp is not None')

    # broadcast arrays
    if zp is None:
        time_or_phase, band = np.broadcast_arrays(time_or_phase, band)
    else:
        time_or_phase, band, zp, zpsys = \
            np.broadcast_arrays(time_or_phase, band, zp, zpsys)

    # convert all to 1d arrays
    ndim = time_or_phase.ndim # save input ndim for return val
    time_or_phase = np.atleast_1d(time_or_phase)
    band = np.atleast_1d(band)
    if zp is not None:
        zp = np.atleast_1d(zp)
        zpsys = np.atleast_1d(zpsys)

    # initialize output arrays
    bandflux = np.zeros(time_or_phase.shape, dtype=np.float)

    # Loop over unique bands.
    for b in set(band):
        mask = band == b
        b = get_bandpass(b)

        # Raise an exception if bandpass is out of model range.
        if (b.wave[0] < model.minwave or b.wave[-1] > model.maxwave):
            raise ValueError(
                'bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                'outside spectral range [{3:.6g}, .., {4:.6g}]'
                .format(b.name, b.wave[0], b.wave[-1], 
                        model.minwave, model.maxwave))

        # Get the flux
        f = model._flux(time_or_phase[mask], b.wave)
        fsum = np.sum(f * b.trans * b.wave * b.dwave, axis=1) / HC_ERG_AA

        if zp is not None:
            zpnorm = 10.**(0.4 * zp[mask])
            bandzpsys = zpsys[mask]
            for ms in set(bandzpsys):
                mask2 = bandzpsys == ms
                ms = get_magsystem(ms)
                zpnorm[mask2] = zpnorm[mask2] / ms.zpbandflux(b)
            fsum *= zpnorm

        bandflux[mask] = fsum

    if ndim == 0:
        return bandflux[0]
    return bandflux

def _bandmag(model, band, magsys, time_or_phase):
    """Support function for bandflux in SourceModel and ObsModel.
    This is necessary to have outside the models because ``phase`` is used in
    SourceModel and ``time`` is used in ObsModel.
    """
    bandflux = _bandflux(model, band, time_or_phase, None, None)
    band, magsys, bandflux = np.broadcast_arrays(band, magsys, bandflux)
    return_scalar = (band.ndim == 0)
    band = band.ravel()
    magsys = magsys.ravel()
    bandflux = bandflux.ravel()

    result = np.empty(bandflux.shape, dtype=np.float)
    for i, (b, ms, f) in enumerate(zip(band, magsys, bandflux)):
        ms = get_magsystem(ms)
        zpf = ms.zpbandflux(b)
        result[i] = -2.5 * np.log10(f / zpf)

    if return_scalar:
        return result[0]
    return result


class _ModelBase(object):
    """Base class for anything with parameters.

    Derived classes must have properties ``_param_names`` (list of str)
    and ``_parameters`` (1-d numpy.ndarray). In the future this might
    use model classes in astropy.modeling as a base class.
    """

    @property
    def param_names(self):
        """List of parameter names."""
        return self._param_names

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        value = np.asarray(value)
        if value.shape != self._parameters.shape:
            raise ValueError("Incorrect number of parameters.")
        self._parameters[:] = value

    def set(self, **param_dict):
        """Set parameters of the model by name."""
        for key, val in param_dict.items():
            try:
                i = self._param_names.index(key)
            except ValueError:
                raise KeyError("Unknown parameter: " + repr(key))
            self._parameters[i] = val

    def __getitem__(self, name):
        try:
            i = self._param_names.index(name)
        except ValueError:
            raise KeyError("Model has no parameter " + repr(name))
        return self._parameters[i]

    def __setitem__(self, name, value):
        try:
            i = self._param_names.index(name)
        except ValueError:
            raise KeyError("Model has no parameter " + repr(name))
        self._parameters[i] = value

    def summary(self):
        return ''

    def __str__(self):
        parameter_lines = [self.summary(), 'parameters:']
        if len(self._param_names) > 0:
            m = max(map(len, self._param_names))
            extralines = ['  ' + k.ljust(m) + ' = ' + repr(v)
                          for k, v in zip(self._param_names, self._parameters)]
            parameter_lines.extend(extralines)
        return '\n'.join(parameter_lines)

    def __copy__(self):
        """Like a normal shallow copy, but makes an actual copy of the
        parameter array."""
        new_model = self.__new__(self.__class__)
        for key, val in self.__dict__.items():
            new_model.__dict__[key] = val
        new_model._parameters = self._parameters.copy()
        return new_model


class SourceModel(_ModelBase):
    """An abstract base class for transient models.
    
    A "transient model" in this case is the spectral time evolution
    of a source as a function of an arbitrary number of parameters.

    This is an abstract base class -- You can't create instances of
    this class. Instead, you must work with subclasses such as
    `TimeSeriesModel`. Subclasses must define (at minimum): 

    * `__init__()`
    * `_param_names` (list of str)
    * `_parameters` (`numpy.ndarray`)
    * `_phase` or `phases` property (`numpy.ndarray`)
    * `_wave` (`numpy.ndarray`)
    * `_flux(ndarray, ndarray)` 
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def phases(self):
        return self._phase

    @property
    def minphase(self):
        return self._phase[0]

    @property
    def maxphase(self):
        return self._phase[-1]

    @property
    def wavelengths(self):
        return self._wave

    @property
    def minwave(self):
        return self._wave[0]

    @property
    def maxwave(self):
        return self._wave[-1]

    @abc.abstractmethod
    def _flux(self, phase, wave):
        pass

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
            phase = self.phases
        else:
            phase = np.asarray(phase)

        if wave is None:
            wave = self._wave
        else:
            wave = np.asarray(wave)

        if np.any(wave < self.minwave) or np.any(wave > self.maxwave):
            raise ValueError('requested wavelength value(s) outside '
                             'model range')

        f = self._flux(phase, wave)

        if phase.ndim == 0:
            if wave.ndim == 0:
                return f[0, 0]
            return f[0, :]
        return f

    def bandflux(self, band, phase=None, zp=None, zpsys=None):
        """Flux through the given bandpass(es) at the given phase(s).

        Default return value is flux in photons / s / cm^2. If zp and zpsys
        are given, flux(es) are scaled to the requested zeropoints.

        Parameters
        ----------
        band : `~sncosmo.Bandpass` or str or list_like
            Bandpass(es) or name(s) of bandpass(es) in registry.
        phase : float or list_like, optional
            Phase(s) in days. Default is `None`, which corresponds to the full
            native phase sampling of the model.
        zp : float or list_like, optional
            If given, zeropoint to scale flux to. If `None` (default) flux
            is not scaled.
        zpsys : `~sncosmo.MagSystem` or str (or list_like), optional
            Determines the magnitude system of the requested zeropoint.
            Cannot be `None` if `zp` is not `None`.

        Returns
        -------
        bandflux : float or `~numpy.ndarray`
            Flux in photons / s /cm^2, unless `zp` and `zpsys` are
            given, in which case flux is scaled so that it corresponds
            to the requested zeropoint. Return value is `float` if all
            input parameters are scalars, `~numpy.ndarray` otherwise.
        """

        if phase is None:
            phase = self.phases
        return _bandflux(self, band, phase, zp, zpsys)


    def bandmag(self, band, magsys, phase=None):
        """Magnitude at the given phase(s) through the given 
        bandpass(es), and for the given magnitude system(s).

        Parameters
        ----------
        phase : float or list_like
            Phase(s) in days.
        band : `Bandpass` or str (or list_like)
            Bandpass or name of bandpass in registry.
        magsys : `MagSystem` or str (or list_like)
            MagSystem or name of MagSystem in registry.

        Returns
        -------
        mag : float or `~numpy.ndarray`
            Magnitude for each item in band, magsys, phase.
            The return value is a float if all parameters are not interables.
            The return value is an `~numpy.ndarray` if any are interable.
        """
        if phase is None:
            phase = self.phases
        return _bandmag(self, band, magsys, phase)

    def peakphase(self, band):
        """Determine phase of maximum flux in the given band.
        
        This method generates the light curve in the given band and
        finds the highest-flux point. It then finds the parabola that
        passes through this point and the two neighboring points, and
        returns the position of the peak of the parabola.
        """

        phases = self.phases
        fluxes = self.bandflux(band)
        i = np.argmax(fluxes)
        if (i == 0) or (i == len(phases) - 1):
            return phases[i]

        x = phases[i-1: i+2]
        y = fluxes[i-1: i+2]
        A = np.hstack([x.reshape(3,1)**2, x.reshape(3,1), np.ones((3,1))])
        a, b, c = np.linalg.solve(A, y)
        return -b / (2 * a)

    def peakmag(self, band, magsys):
        """Calculate peak apparent magnitude in rest-frame bandpass."""

        peakphase = self.peakphase(band)
        return self.bandmag(band, magsys, phase=peakphase)

    def set_peakmag(self, m, band, magsys):
        """Set peak apparent magnitude in rest-frame bandpass."""

        m_current = self.peakmag(band, magsys)
        factor = 10.**(0.4 * (m_current - m))
        self._parameters[0] = factor * self._parameters[0]

    def __repr__(self):
        name = ''
        version = ''
        if self.name is not None:
            name = ' {0!r:s}'.format(self.name)
        if self.version is not None:
            version = ' version={0!r:s}'.format(self.version)
        return "<{0:s}{1:s}{2:s} at 0x{3:x}>".format(
            self.__class__.__name__, name, version, id(self))

    def summary(self):
        summary = """\
        class      : {}
        name       : {}
        version    : {}
        phases     : [{:.6g}, .., {:.6g}] days ({:d} points)
        wavelengths: [{:.6g}, .., {:.6g}] Angstroms ({:d} points)"""\
        .format(
            self.__class__.__name__, self.name, self.version,
            self.minphase, self.maxphase, len(self._phase),
            self.minwave, self.maxwave, len(self._wave)
            )
        return dedent(summary)


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

    _param_names = ['amplitude']
    param_names_latex = ['A']
    param_bounds = [(0., None)]

    def __init__(self, phase, wave, flux,
                 name=None, version=None):

        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._parameters = np.array([1.])
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

    _param_names = ['amplitude', 's']
    param_names_latex = ['A', 's']
    param_bounds = [(0., None), (0., None)]

    def __init__(self, phase, wave, flux, name=None, version=None):
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._parameters = np.array([1., 1.])
        self._model_flux = Spline2d(phase, wave, flux, kx=2, ky=2)

    @property
    def phases(self):
        return self._parameters[1] * self._phase

    @property
    def minphase(self):
        return self._parameters[1] * self._phase[0]

    @property
    def maxphase(self):
        return self._parameters[1] * self._phase[-1]

    def _flux(self, phase, wave):
        return (self._parameters[0] *
                self._model_flux(phase / self._parameters[1], wave))


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

    _param_names = ['x0', 'x1', 'c']
    param_names_latex = ['x_0', 'x_1', 'c']
    param_bounds = [(0., None), (None, None), (None, None)]

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
        self._parameters = np.array([1., 0., 0.])
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
            values = 10.**(-12. + 0.27) * values
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
        self._model['clbase'] = Spline1d(self._wave, clbase, k=2)

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

    def colorlaw(self, wave=None):
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
        calculating the model flux, a spline fit to this function is
        used rather than the function itself. Therefore this will not be
        *exactly* equivalent to the color law used when evaluating the model
        flux for arbitrary wavelengths. However, it will be exactly
        equivalent at the native model wavelengths.
        """
        if wave is None:
            wave = self._wave
        else:
            wave = np.asarray(wave)
        if wave.ndim == 0:
            return self._colorlaw(np.ravel(wave))[0]
        else:
            return self._colorlaw(wave)


class ObsModel(_ModelBase):
    """An observer-frame model.

    Parameters
    ----------
    source_model : `~sncosmo.SourceModel`
        The model for the spectral evolution of the source.
    effects : list of `~sncosmo.PropagationEffect`s
    effect_names : list of str
    effect_frames : list of str

    Notes
    -----
    SourceModel and PropagationEffects are copied upon instanciation.
    """

    def __init__(self, source_model, effects=None,
                 effect_names=None, effect_frames=None):
        self._param_names = ['z', 't0']
        self.param_names_latex = ['z', 't_0']
        self.param_bounds = [(None, None), (None, None)]
        self._parameters = np.array([0., 0.])
        self._source = cp(source_model)
        self.name = self._source.name  # TODO be more careful with the name
        self._effects = []
        self._effect_names = []
        self._effect_frames = []
        self._synchronize_parameters()

        # Add PropagationEffects
        if (effects is not None or effect_names is not None or
            effect_frames is not None):
            try:
                same_length = (len(effects) == len(effect_names) and
                               len(effects) == len(effect_frames))
            except TypeError:
                raise ValueError('effects, effect_names, and effect_values '
                                 'should all be iterables.')
            if not same_length:
                raise ValueError('effects, effect_names and effect_values '
                                 'must have matching lengths')

            for effect, name, frame in zip(effects, effect_names,
                                           effect_frames): 
                self.add_effect(effect, name, frame)

    # TODO: Check if PropagationEffect covers complete range of model
    def add_effect(self, effect, name, frame):
        """
        Add a PropagationEffect to the model.

        Parameters
        ----------
        name : str
            Name of the effect.
        effect : `~sncosmo.PropagationEffect`
            Propagation effect.
        frame : {'rest', 'obs'}
        """
        if not isinstance(effect, PropagationEffect):
            raise ValueError('effect is not a PropagationEffect')
        if frame not in ['rest', 'obs']:
            raise ValueError("frame must be one of: 'rest', 'obs'")
        self._effects.append(cp(effect))
        self._effect_names.append(name)
        self._effect_frames.append(frame)
        self._synchronize_parameters()

    @property
    def source(self):
        """The SourceModel instance. Use to set source apparent mag."""
        return self._source

    @property
    def effect_names(self):
        """Names of propagation effects."""
        return self._effect_names

    @property
    def effects(self):
        """Dictionary of PropagationEffect instances."""
        return odict(zip(self._effect_names, self._effects))

    def _synchronize_parameters(self):
        """Synchronize parameter names and parameter arrays between
        the aggregated parameters and those of the individual source and
        effects.
        """

        # Build a new list of parameter names
        self._param_names = self._param_names[0:2]
        self._param_names.extend(self._source.param_names)
        for effect, effect_name in zip(self._effects, self._effect_names):
            self._param_names.extend([effect_name + param_name
                                      for param_name in effect.param_names])

        # Build a new list of latex parameter names
        self.param_names_latex = self.param_names_latex[0:2]
        self.param_names_latex.extend(self._source.param_names_latex)
        for effect, effect_name in zip(self._effects, self._effect_names):
            self.param_names_latex.extend(
                [name + '_{' + effect_name + '}' 
                 for name in effect.param_names_latex])

        # Build a new list of parameter bounds
        self.param_bounds = self.param_bounds[0:2]
        self.param_bounds.extend(self._source.param_bounds)
        for effect in self._effects:
            self.param_bounds.extend(effect.param_bounds)

        # For each "model", get its parameter array.
        param_arrays = [self._parameters[0:2]]
        models = [self._source] + self._effects
        param_arrays.extend([m._parameters for m in models])
        
        # Create a new parameter array built from the individual arrays
        # and reference the individual parameter arrays to the new combined
        # array.
        self._parameters = np.concatenate(param_arrays)
        pos = 2
        for m in models:
            l = len(m._parameters)
            m._parameters = self._parameters[pos:pos+l]
            pos += l

    # ----------------------------------------------------------------
    # Time and wavelength grid

    @property
    def times(self):
        """Native times at which the source model is sampled."""
        return (self._parameters[1] +
                (1. + self._parameters[0]) * self._source.phases)

    @property
    def mintime(self):
        """Minimum time at which the model is defined."""
        return (self._parameters[1] +
                (1. + self._parameters[0]) * self._source.minphase)
    @property
    def maxtime(self):
        """Maximum time at which the model is defined."""
        return (self._parameters[1] +
                (1. + self._parameters[0]) * self._source.maxphase)

    @property
    def wavelengths(self):
        """Native observer-frame wavelengths at which the model is sampled."""
        return self._source.wavelengths * (1. + self._parameters[0])

    @property
    def minwave(self):
        """Minimum observer-frame wavelength of the model."""
        return self._source.minwave * (1. + self._parameters[0])

    @property
    def maxwave(self):
        """Maximum observer-frame wavelength of the model."""
        return self._source.maxwave * (1. + self._parameters[0])

    # ----------------------------------------------------------------
    # Flux
    
    def _flux(self, time, wave):
        """Array flux function."""
        a = 1. / (1. + self._parameters[0])
        phase = (time - self._parameters[1]) * a
        restwave = wave * a

        # Note that below we multiply by the scale factor to conserve
        # bolometric luminosity.
        f = a * self._source._flux(phase, restwave)

        # Pass the flux through the PropagationEffects.
        for effect, frame in zip(self._effects, self._effect_frames):
            if frame == 'obs':
                f = effect.propagate(wave, f)
            else:
                f = effect.propagate(restwave, f)

        return f

    def flux(self, time=None, wave=None):
        """The spectral flux density at the given time and wavelength values.

        Parameters
        ----------
        time : float or list_like, optional
            Time(s) in days. If `None` (default), the times corresponding
            to the native phases of the model are used.
        wave : float or list_like, optional
            Wavelength(s) in Angstroms. If `None` (default), the native
            wavelengths of the model are used.

        Returns
        -------
        flux : float or `~numpy.ndarray`
            Spectral flux density values in ergs / s / cm^2 / Angstrom.
        """
        
        # Defaults
        if time is None:
            time = self.times
        else:
            time = np.asarray(time)
        if wave is None:
            wave = self.wavelengths
        else:
            wave = np.asarray(wave)

        # Check wavelength values
        if np.any(wave < self.minwave) or np.any(wave > self.maxwave):
            raise ValueError('requested wavelength value(s) outside '
                             'model range')

        # Get the flux
        f = self._flux(time, wave)

        # Return array according to dimension of inputs.
        if np.isscalar(time) or time.ndim == 0:
            if np.isscalar(wave) or wave.ndim == 0:
                return f[0, 0]
            return f[0, :]
        return f

    # ----------------------------------------------------------------------
    # Bandpass-related functions

    def bandoverlap(self, band, z=None):
        """Return True if model dispersion range fully overlaps the band.

        Parameters
        ----------
        band : `~sncosmo.Bandpass`, str or list_like
            Bandpass, name of bandpass in registry, or list or array thereof.
        z : float or list_like, optional
            If given, evaluate the overlap when the model is at the given
            redshifts. If `None`, use the model redshift.

        Returns
        -------
        overlap : bool or `~numpy.ndarray`
            
        """
        
        band = np.asarray(band)
        if z is None:
            z = self._parameters[0]
        z = np.asarray(z)
        ndim = (band.ndim, z.ndim)
        band = band.ravel()
        z = z.ravel()
        overlap = np.empty((len(band), len(z)), dtype=np.bool)
        for i, b in enumerate(band):
            b = get_bandpass(b)
            overlap[i, :] = ((b.wave[0] > self._source.minwave * (1. + z)) &
                             (b.wave[-1] < self._source.maxwave * (1. + z)))
        if ndim == (0, 0):
            return overlap[0, 0]
        if ndim[1] == 0:
            return overlap[:, 0]
        return overlap

    def bandflux(self, band, time=None, zp=None, zpsys=None):
        """Flux through the given bandpass(es) at the given time(s).

        Default return value is flux in photons / s / cm^2. If zp and zpsys
        are given, flux(es) are scaled to the requested zeropoints.

        Parameters
        ----------
        band : `~sncosmo.Bandpass` or str or list_like
            Bandpass(es) or name(s) of bandpass(es) in registry.
        time : float or list_like, optional
            Time(s) in days. Default is `None`, which corresponds to the full
            native time sampling of the model.
        zp : float or list_like, optional
            If given, zeropoint to scale flux to. If `None` (default) flux
            is not scaled.
        zpsys : `~sncosmo.MagSystem` or str (or list_like), optional
            Determines the magnitude system of the requested zeropoint.
            Cannot be `None` if `zp` is not `None`.

        Returns
        -------
        bandflux : float or `~numpy.ndarray`
            Flux in photons / s /cm^2, unless `zp` and `zpsys` are
            given, in which case flux is scaled so that it corresponds
            to the requested zeropoint. Return value is `float` if all
            input parameters are scalars, `~numpy.ndarray` otherwise.
        """

        if time is None:
            time = self.times
        return _bandflux(self, band, time, zp, zpsys)

    def bandmag(self, band, magsys, time=None):
        """Magnitude at the given time(s) through the given 
        bandpass(es), and for the given magnitude system(s).

        Parameters
        ----------
        time : float or list_like
            Time(s) in days.
        band : `Bandpass` or str (or list_like)
            Bandpass or name of bandpass in registry.
        magsys : `MagSystem` or str (or list_like)
            MagSystem or name of MagSystem in registry.

        Returns
        -------
        mag : float or `~numpy.ndarray`
            Magnitude for each item in time, band, magsys.
            The return value is a float if all parameters are not interables.
            The return value is an `~numpy.ndarray` if any are interable.
        """
        if time is None:
            time = self.times
        return _bandmag(self, band, magsys, time)

    def source_peakabsmag(self, band, magsys, cosmo=cosmology.WMAP9):
        return (self._source.peakmag(band, magsys) -
                cosmo.distmod(self._parameters[0]).value)

    def set_source_peakabsmag(self, absmag, band, magsys,
                              cosmo=cosmology.WMAP9):
        if self._parameters[0] <= 0.:
            raise ValueError('absolute magnitude undefined when z<=0.')
        m = absmag + cosmo.distmod(self._parameters[0]).value
        self._source.set_peakmag(m, band, magsys)

    def summary(self):
        head = "<{0:s} at 0x{1:x}>".format(self.__class__.__name__, id(self))
        s = 'source:\n' + self._source.summary()
        summaries = [head, s.replace('\n', '\n  ')]
        for effect, name, frame in zip(self._effects,
                                       self._effect_names,
                                       self._effect_frames):
            s = ('effect (name={} frame={}):\n{}'
                 .format(repr(name), repr(frame), effect.summary()))
            summaries.append(s.replace('\n', '\n  '))
        return '\n'.join(summaries)

    def __copy__(self):
        new = ObsModel(self._source,
                       effects=self._effects,
                       effect_names=self._effect_names,
                       effect_frames=self._effect_frames)
        new._parameters[0:2] = self._parameters[0:2]
        return new


class PropagationEffect(_ModelBase):
    """Base class for propagation effects."""
    
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def propagate(self, wave, flux):
        pass

class InterpolatedRvDust(PropagationEffect):
    """Dust propagation effect. Wraps extinction functions.

    Parameters
    ----------
    model : str, optional
        The dust extinction model to use.

    See Also
    --------
    extinction
    """

    _param_names = ['ebv']
    param_names_latex = ['E(B-V)']
    param_bounds = [(None, None)]

    def __init__(self, model='f99', ebv=0., r_v=3.1, minwave=1000.,
                 maxwave=30000., spline_points=2000):
        self._parameters = np.array([ebv])
        self.minwave = minwave
        self.maxwave = maxwave
        self._wave = np.logspace(np.log10(minwave), np.log10(maxwave),
                                 spline_points)
        self._model = model
        self.r_v = r_v

    @property
    def r_v(self):
        """R_V value"""
        return self._r_v

    @r_v.setter
    def r_v(self, value):
        self._r_v = value

        # Recalculate spline
        trans_base = 10.**(-0.4 * extinction(self._wave, ebv=1.,
                                             r_v=value, model=self._model))
        self._spline = Spline1d(self._wave, trans_base)

    def propagate(self, wave, flux):
        """Propagate the flux."""
        return flux * self._spline(wave) ** self._parameters[0]

    def __repr__(self):
        return "{}(model={}, ebv={}, r_v={}, minwave={}, maxwave={}, " + \
            "spline_points={})".format(
            self.__class__.__name__, repr(self._model),
            self._parameters[0], self._r_v,
            self._wave[0], self._wave[-1], len(self._wave))

    def summary(self):
        summary = """\
        class      : {}
        model      : {}
        r_v        : {}
        wavelengths: [{:.6g}, .., {:.6g}] Angstroms ({:d} points)"""\
        .format(
            self.__class__.__name__, repr(self._model), self._r_v,
            self.minwave, self.maxwave, len(self._wave)
            )
        return dedent(summary)
