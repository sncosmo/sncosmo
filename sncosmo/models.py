# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Classes to represent spectral models of astronomical transients."""

import abc
import os
from copy import copy as cp
from math import ceil
from textwrap import dedent

import extinction
import numpy as np
from astropy import (cosmology, units as u)
from astropy.utils.misc import isiterable
from scipy.interpolate import (
    InterpolatedUnivariateSpline as Spline1d,
    RectBivariateSpline as Spline2d
)

from ._registry import Registry
from .bandpasses import Bandpass, get_bandpass
from .constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING
from .io import (
    read_griddata_ascii, read_griddata_fits,
    read_multivector_griddata_ascii
)
from .magsystems import get_magsystem
from .salt2utils import BicubicInterpolator, SALT2ColorLaw
from .utils import integration_grid

__all__ = ['get_source', 'Source', 'TimeSeriesSource', 'StretchSource',
           'SUGARSource', 'SALT2Source', 'SALT3Source', 'MLCS2k2Source',
           'SNEMOSource', 'Model', 'PropagationEffect', 'CCM89Dust',
           'OD94Dust', 'F99Dust']

_SOURCES = Registry()


def _check_for_fitpack_error(e, a, name):
    """Raise a more informative error message for fitpack errors.

    This is implemented as a separate function rather than subclassing Spline2d
    so that we can raise the error closer to user-facing functions. We
    may wish to change this behavior in the future. For example, if some
    models are implemented not based on RectBivariateSpline so that they
    don't have this restriction.

    Parameters
    ----------
    e : ValueError
    a : `~numpy.ndarray` (0-d or 1-d)
    name : str
    """

    # Check if the error is a specific one raise by RectBivariateSpline
    # If it is, check if supplied array is *not* monotonically increasing
    if (len(e.args) > 0 and
            e.args[0].startswith("Error code returned by bispev: 10") and
            np.any(np.ediff1d(a) < 0.)):
        raise ValueError(name + ' must be monotonically increasing')


def get_source(name, version=None, copy=False):
    """Retrieve a Source from the registry by name.

    Parameters
    ----------
    name : str
        Name of source in the registry.
    version : str, optional
        Version identifier for sources with multiple versions. Default is
        `None` which corresponds to the latest, or only, version.
    copy : bool, optional
        If True and if `name` is already a Source instance, return a copy of
        it. (If `name` is a str a copy of the instance
        in the registry is always returned, regardless of the value of this
        parameter.) Default is False.
    """

    # If we need to retrieve from the registry, we want to return a shallow
    # copy, in order to keep the copy in the registry "pristene". However, we
    # *don't* want a shallow copy otherwise. Therefore,
    # we need to check if `name` is already an instance of Model before
    # going to the registry, so we know whether or not to make a shallow copy.
    if isinstance(name, Source):
        if copy:
            return cp(name)
        else:
            return name
    else:
        return cp(_SOURCES.retrieve(name, version=version))


def _bandflux_single(model, band, time_or_phase):
    """Synthetic photometry of model through a single bandpass.

    Parameters
    ----------
    model : Source or Model
    band : Bandpass
    time_or_phase : `~numpy.ndarray` (1-d)
    """

    # Check that bandpass wavelength range is fully contained in model
    # wavelength range.
    if (band.minwave() < model.minwave() or band.maxwave() > model.maxwave()):
        raise ValueError('bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                         'outside spectral range [{3:.6g}, .., {4:.6g}]'
                         .format(band.name, band.minwave(), band.maxwave(),
                                 model.minwave(), model.maxwave()))

    # Set up wavelength grid. Spacing (dwave) evenly divides the bandpass,
    # closest to 5 angstroms without going over.
    wave, dwave = integration_grid(band.minwave(), band.maxwave(),
                                   MODEL_BANDFLUX_SPACING)
    trans = band(wave)
    f = model._flux(time_or_phase, wave)

    return np.sum(wave * trans * f, axis=1) * dwave / HC_ERG_AA


def _bandflux(model, band, time_or_phase, zp, zpsys):
    """Support function for bandflux in Source and Model.
    This is necessary to have outside because ``phase`` is used in Source
    and ``time`` is used in Model, and we want the method signatures to
    have the right variable name.
    """

    if zp is not None and zpsys is None:
        raise ValueError('zpsys must be given if zp is not None')

    # broadcast arrays
    if zp is None:
        time_or_phase, band = np.broadcast_arrays(time_or_phase, band)
    else:
        time_or_phase, band, zp, zpsys = \
            np.broadcast_arrays(time_or_phase, band, zp, zpsys)

    # Convert all to 1-d arrays.
    ndim = time_or_phase.ndim  # Save input ndim for return val.
    time_or_phase = np.atleast_1d(time_or_phase)
    band = np.atleast_1d(band)
    if zp is not None:
        zp = np.atleast_1d(zp)
        zpsys = np.atleast_1d(zpsys)

    # initialize output arrays
    bandflux = np.zeros(time_or_phase.shape, dtype=float)

    # Loop over unique bands.
    for b in set(band):
        mask = band == b
        b = get_bandpass(b)

        fsum = _bandflux_single(model, b, time_or_phase[mask])

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
    """Support function for bandflux in Source and Model.
    This is necessary to have outside the models because ``phase`` is used in
    Source and ``time`` is used in Model.
    """
    bandflux = _bandflux(model, band, time_or_phase, None, None)
    band, magsys, bandflux = np.broadcast_arrays(band, magsys, bandflux)
    return_scalar = (band.ndim == 0)
    band = band.ravel()
    magsys = magsys.ravel()
    bandflux = bandflux.ravel()

    result = np.empty(bandflux.shape, dtype=float)
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
    and ``_parameters`` (1-d numpy.ndarray).
    """

    @property
    def param_names(self):
        """List of parameter names."""
        return self._param_names

    @property
    def parameters(self):
        """Parameter value array"""
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        value = np.asarray(value)
        if value.shape != self._parameters.shape:
            raise ValueError("Incorrect number of parameters.")
        self._parameters[:] = value

    def set(self, **param_dict):
        """Set parameters of the model by name."""
        self.update(param_dict)

    def update(self, param_dict):
        """Set parameters of the model from a dictionary."""
        for key, value in param_dict.items():
            self[key] = value

    def __setitem__(self, key, value):
        """Set a single parameter of the model by name."""
        try:
            i = self._param_names.index(key)
        except ValueError:
            raise KeyError("Unknown parameter: " + repr(key))
        self._parameters[i] = value

    def get(self, name):
        """Get parameter of the model by name."""
        return self[name]

    def __getitem__(self, name):
        """Get parameter of the model by name"""
        try:
            i = self._param_names.index(name)
        except ValueError:
            raise KeyError("Model has no parameter " + repr(name))
        return self._parameters[i]

    def _headsummary(self):
        return ''

    def __str__(self):
        parameter_lines = [self._headsummary(), 'parameters:']
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


class Source(_ModelBase):
    """An abstract base class for transient models.

    A "transient model" in this case is the spectral time evolution
    of a source, as defined in the rest-frame of the transient: ``Source``
    subclass instances define a spectral flux density
    (in, e.g., erg / s / cm^2 / Angstrom) as a function of phase and
    wavelength, where phase and wavelength are in the source's rest-frame.
    (The ``Model`` class wraps a ``Source`` instance and takes care of
    redshift and time dilation.) This two-dimensional spectral surface
    can be a function of any number of parameters that alter its amplitude
    or shape. Different subclasses will have different parameters.

    This is an abstract base class -- You can't create instances of
    this class. Instead, you must work with subclasses such as
    ``TimeSeriesSource``. Subclasses must define (at minimum):

    * `__init__()`
    * `_param_names` (list of str)
    * `_parameters` (`numpy.ndarray`)
    * `_flux(ndarray, ndarray)`
    * `minphase()`
    * `maxphase()`
    * `minwave()`
    * `maxwave()`
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    def minphase(self):
        return self._phase[0]

    def maxphase(self):
        return self._phase[-1]

    def minwave(self):
        return self._wave[0]

    def maxwave(self):
        return self._wave[-1]

    @abc.abstractmethod
    def _flux(self, phase, wave):
        pass

    def flux(self, phase, wave):
        """The spectral flux density at the given phase and wavelength values.

        Parameters
        ----------
        phase : float or list_like, optional
            Phase(s) in days. Must be monotonically increasing.
            If `None` (default), the native phases of the model are used.
        wave : float or list_like, optional
            Wavelength(s) in Angstroms. Must be monotonically increasing.
            If `None` (default), the native wavelengths of the model are used.

        Returns
        -------
        flux : float or `~numpy.ndarray`
            Spectral flux density values in ergs / s / cm^2 / Angstrom.
        """
        phase = np.asarray(phase)
        wave = np.asarray(wave)
        if np.any(wave < self.minwave()) or np.any(wave > self.maxwave()):
            raise ValueError('requested wavelength value(s) outside '
                             'model range')
        try:
            f = self._flux(phase, wave)
        except ValueError as e:
            _check_for_fitpack_error(e, phase, 'phase')
            _check_for_fitpack_error(e, wave, 'wave')
            raise e

        if phase.ndim == 0:
            if wave.ndim == 0:
                return f[0, 0]
            return f[0, :]
        return f

    def bandflux(self, band, phase, zp=None, zpsys=None):
        """Flux through the given bandpass(es) at the given phase(s).

        Default return value is flux in photons / s / cm^2. If zp and zpsys
        are given, flux(es) are scaled to the requested zeropoints.

        Parameters
        ----------
        band : str or list_like
            Name(s) of bandpass(es) in registry.
        phase : float or list_like, optional
            Phase(s) in days. Default is `None`, which corresponds to the full
            native phase sampling of the model.
        zp : float or list_like, optional
            If given, zeropoint to scale flux to (must also supply ``zpsys``).
            If not given, flux is not scaled.
        zpsys : str or list_like, optional
            Name of a magnitude system in the registry, specifying the system
            that ``zp`` is in.

        Returns
        -------
        bandflux : float or `~numpy.ndarray`
            Flux in photons / s /cm^2, unless `zp` and `zpsys` are
            given, in which case flux is scaled so that it corresponds
            to the requested zeropoint. Return value is `float` if all
            input parameters are scalars, `~numpy.ndarray` otherwise.
        """
        try:
            return _bandflux(self, band, phase, zp, zpsys)
        except ValueError as e:
            _check_for_fitpack_error(e, phase, 'phase')
            raise e

    def bandmag(self, band, magsys, phase):
        """Magnitude at the given phase(s) through the given
        bandpass(es), and for the given magnitude system(s).

        Parameters
        ----------
        band : str or list_like
            Name(s) of bandpass in registry.
        magsys : str or list_like
            Name(s) of `~sncosmo.MagSystem` in registry.
        phase : float or list_like
            Phase(s) in days.

        Returns
        -------
        mag : float or `~numpy.ndarray`
            Magnitude for each item in band, magsys, phase.
            The return value is a float if all parameters are not iterables.
            The return value is an `~numpy.ndarray` if any are iterable.
        """
        return _bandmag(self, band, magsys, phase)

    def peakphase(self, band_or_wave, sampling=1.):
        """Determine phase of maximum flux for the given band/wavelength.

        This method generates the light curve in the given band/wavelength and
        finds the highest-flux point. It then finds the parabola that
        passes through this point and the two neighboring points, and
        returns the position of the peak of the parabola.
        """

        # Array of phases to sample at.
        nsamples = int(ceil((self.maxphase()-self.minphase()) / sampling)) + 1
        phases = np.linspace(self.minphase(), self.maxphase(), nsamples)

        if isinstance(band_or_wave, (str, Bandpass)):
            fluxes = self.bandflux(band_or_wave, phases)
        else:
            fluxes = self.flux(phases, band_or_wave)[:, 0]

        i = np.argmax(fluxes)
        if (i == 0) or (i == len(phases) - 1):
            return phases[i]

        x = phases[i-1: i+2]
        y = fluxes[i-1: i+2]
        A = np.hstack([x.reshape(3, 1)**2, x.reshape(3, 1), np.ones((3, 1))])
        a, b, c = np.linalg.solve(A, y)
        return -b / (2 * a)

    def peakmag(self, band, magsys, sampling=1.0):
        """Peak apparent magnitude in rest-frame bandpass."""

        peakphase = self.peakphase(band, sampling=sampling)
        return self.bandmag(band, magsys, peakphase)

    def set_peakmag(self, m, band, magsys, sampling=1.0):
        """Set peak apparent magnitude in rest-frame bandpass."""

        m_current = self.peakmag(band, magsys, sampling=sampling)
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

    def _headsummary(self):
        summary = """\
        class      : {0}
        name       : {1!r}
        version    : {2}
        phases     : [{3:.6g}, .., {4:.6g}] days
        wavelengths: [{5:.6g}, .., {6:.6g}] Angstroms"""\
        .format(
            self.__class__.__name__, self.name, self.version,
            self.minphase(), self.maxphase(),
            self.minwave(), self.maxwave())
        return dedent(summary)


class TimeSeriesSource(Source):
    """A single-component spectral time series model.

    The spectral flux density of this model is given by

    .. math::

       F(t, \\lambda) = A \\times M(t, \\lambda)

    where _M_ is the flux defined on a grid in phase and wavelength
    and _A_ (amplitude) is the single free parameter of the model. The
    amplitude _A_ is a simple unitless scaling factor applied to
    whatever flux values are used to initialize the
    ``TimeSeriesSource``. Therefore, the _A_ parameter has no
    intrinsic meaning. It can only be interpreted in conjunction with
    the model values. Thus, it is meaningless to compare the _A_
    parameter between two different ``TimeSeriesSource`` instances with
    different model data.

    Parameters
    ----------
    phase : `~numpy.ndarray`
        Phases in days.
    wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
    flux : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape ``(num_phases, num_wave)``.
    zero_before : bool, optional
        If True, flux at phases before minimum phase will be zeroed. The
        default is False, in which case the flux at such phases will be equal
        to the flux at the minimum phase (``flux[0, :]`` in the input array).
    time_spline_degree : int, optional
        Degree of the spline used for interpolation in the time (phase)
        direction. By default this is set to 3 (i.e. cubic spline). For models
        that are defined with sparse time grids this can lead to large
        interpolation uncertainties and negative fluxes. If this is a problem,
        set time_spline_degree to 1 to use linear interpolation instead.
    name : str, optional
        Name of the model. Default is `None`.
    version : str, optional
        Version of the model. Default is `None`.
    """

    _param_names = ['amplitude']
    param_names_latex = ['A']

    def __init__(self, phase, wave, flux, zero_before=False,
                 time_spline_degree=3, name=None, version=None):
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._parameters = np.array([1.])
        self._model_flux = Spline2d(phase, wave, flux, kx=time_spline_degree,
                                    ky=3)
        self._zero_before = zero_before

    def _flux(self, phase, wave):
        f = self._parameters[0] * self._model_flux(phase, wave)
        if self._zero_before:
            mask = np.atleast_1d(phase) < self.minphase()
            f[mask, :] = 0.
        return f


class StretchSource(Source):
    """A single-component spectral time series model, that "stretches" in
    time.

    The spectral flux density of this model is given by

    .. math::

       F(t, \\lambda) = A \\times M(t / s, \\lambda)

    where _A_ is the amplitude and _s_ is the "stretch".

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

    def __init__(self, phase, wave, flux, name=None, version=None):
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._parameters = np.array([1., 1.])
        self._model_flux = Spline2d(phase, wave, flux, kx=3, ky=3)

    def minphase(self):
        return self._parameters[1] * self._phase[0]

    def maxphase(self):
        return self._parameters[1] * self._phase[-1]

    def _flux(self, phase, wave):
        return (self._parameters[0] *
                self._model_flux(phase / self._parameters[1], wave))


class SUGARSource(Source):
    """
    The SUGAR Type Ia supernova spectral time series template.

    The spectral energy distribution of this model is given by

    .. math::

        F(t, \\lambda) = q_0 10^{-0.4 (M_0(t, \\lambda)
                                + q_1 \\alpha_1(t, \\lambda)
                                + q_2 \\alpha_2(t, \\lambda)
                                + q_3 \\alpha_3(t, \\lambda)
                                + A_v CCM(\\lambda))}
                                (10^{-3} c\\lambda^{2})

    where ``q_0``, ``q_1``, ``q_2``, ``q_3``,
    and ``A_v`` are the free parameters
    of the model,``alpha_0``, ``alpha_1``,
    `alpha_2``, `alpha_3``, `CCM`` are
    the template vectors of the model.
    The ``q_0`` is the equivalent parameter
    in flux of the ``Delta M_{gray}``
    parameter define in Leget et al. 2020.

    Parameters
    ----------
    modeldir : str, optional
        Directory path containing model component files. Default is `None`,
        which means that no directory is prepended to filenames when
        determining their path.

    m0file : str or fileobj, optional
    alpha1file : str or fileobj, optional
    alpha2file : str or fileobj, optional
    alpha3file : str or fileobj, optional
    CCMfile: str or fileobj, optional
        Filenames of various model components. Defaults are:
        * m0file = 'sugar_template_0.dat' (2-d grid)
        * alpha1file = 'sugar_template_1.dat' (2-d grid)
        * alpha2file = 'sugar_template_2.dat' (2-d grid)
        * alpha3file = 'sugar_template_3.dat' (2-d grid)
        * CCMfile = 'sugar_template_4.dat' (2-d grid)

    Notes
    -----
    The "2-d grid" files have the format ``<phase> <wavelength>
    <value>`` on each line.
    """
    _param_names = ['q0', 'q1', 'q2', 'q3', 'Av']
    param_names_latex = ['q_0', 'q_1', 'q_2', 'q_3', 'A_v']

    def __init__(self, modeldir=None,
                 m0file='sugar_template_0.dat',
                 alpha1file='sugar_template_1.dat',
                 alpha2file='sugar_template_2.dat',
                 alpha3file='sugar_template_3.dat',
                 CCMfile='sugar_template_4.dat',
                 name=None, version=None):

        self.name = name
        self.version = version
        self._model = {}
        self.M_keys = ['M0', 'ALPHA1', 'ALPHA2', 'ALPHA3', 'CCM']
        self._parameters = np.zeros(len(self.M_keys))
        self._parameters[0] = 1e-15
        names_or_objs = {'M0': m0file,
                         'ALPHA1': alpha1file,
                         'ALPHA2': alpha2file,
                         'ALPHA3': alpha3file,
                         'CCM': CCMfile}

        # Make filenames into full paths.
        if modeldir is not None:
            for k in names_or_objs:
                v = names_or_objs[k]
                if (v is not None and isinstance(v, str)):
                    names_or_objs[k] = os.path.join(modeldir, v)

        for i, key in enumerate(self.M_keys):
            phase, wave, values = read_griddata_ascii(names_or_objs[key])
            self._model[key] = BicubicInterpolator(phase, wave, values)
            if key == 'M0':
                # The "native" phases and wavelengths of the model are those
                self._phase = np.linspace(-12., 48, 21)
                self._wave = wave

    def _flux(self, phase, wave):
        mag_sugar = self._model['M0'](phase, wave)
        for i, key in enumerate(self.M_keys):
            if key != 'M0':
                comp = self._model[key](phase, wave) * self._parameters[i]
                mag_sugar += comp
        # Mag AB used in the training of SUGAR.
        mag_sugar += 48.59
        wave_factor = (wave ** 2 / 299792458. * 1.e-10)
        flux = (self._parameters[0] * 10. ** (-0.4 * mag_sugar) / wave_factor)

        if hasattr(phase, '__iter__'):
            not_define = ~((phase > -12) & (phase < 48))
            flux[not_define] = 0
            return flux
        else:
            if phase < -12 or phase > 48:
                return np.zeros_like(wave)
            else:
                return flux


class SALT2Source(Source):
    """The SALT2 Type Ia supernova spectral timeseries model.

    The spectral flux density of this model is given by

    .. math::

       F(t, \\lambda) = x_0 (M_0(t, \\lambda) + x_1 M_1(t, \\lambda))
                       \\times 10^{-0.4 CL(\\lambda) c}

    where ``x0``, ``x1`` and ``c`` are the free parameters of the model,
    ``M_0``, ``M_1`` are the zeroth and first components of the model, and
    ``CL`` is the colorlaw, which gives the extinction in magnitudes for
    ``c=1``.

    Parameters
    ----------
    modeldir : str, optional
        Directory path containing model component files. Default is `None`,
        which means that no directory is prepended to filenames when
        determining their path.
    m0file, m1file, clfile : str or fileobj, optional
        Filenames of various model components. Defaults are:

        * m0file = 'salt2_template_0.dat' (2-d grid)
        * m1file = 'salt2_template_1.dat' (2-d grid)
        * clfile = 'salt2_color_correction.dat'

    errscalefile, lcrv00file, lcrv11file, lcrv01file, cdfile : str or fileobj
        (optional) Filenames of various model components for
        model covariance in synthetic photometry. See
        ``bandflux_rcov`` for details.  Defaults are:

        * errscalefile = 'salt2_lc_dispersion_scaling.dat' (2-d grid)
        * lcrv00file = 'salt2_lc_relative_variance_0.dat' (2-d grid)
        * lcrv11file = 'salt2_lc_relative_variance_1.dat' (2-d grid)
        * lcrv01file = 'salt2_lc_relative_covariance_01.dat' (2-d grid)
        * cdfile = 'salt2_color_dispersion.dat' (1-d grid)

    Notes
    -----
    The "2-d grid" files have the format ``<phase> <wavelength>
    <value>`` on each line.

    The phase and wavelength values of the various components don't
    necessarily need to match. (In the most recent salt2 model data,
    they do not all match.) The phase and wavelength values of the
    first model component (in ``m0file``) are taken as the "native"
    sampling of the model, even though these values might require
    interpolation of the other model components.

    """

    # These files are distributed with SALT2 model data but not currently
    # used:
    # v00file = 'salt2_spec_variance_0.dat'              : 2dgrid
    # v11file = 'salt2_spec_variance_1.dat'              : 2dgrid
    # v01file = 'salt2_spec_covariance_01.dat'           : 2dgrid

    _param_names = ['x0', 'x1', 'c']
    param_names_latex = ['x_0', 'x_1', 'c']
    _SCALE_FACTOR = 1e-12

    def __init__(self, modeldir=None,
                 m0file='salt2_template_0.dat',
                 m1file='salt2_template_1.dat',
                 clfile='salt2_color_correction.dat',
                 cdfile='salt2_color_dispersion.dat',
                 errscalefile='salt2_lc_dispersion_scaling.dat',
                 lcrv00file='salt2_lc_relative_variance_0.dat',
                 lcrv11file='salt2_lc_relative_variance_1.dat',
                 lcrv01file='salt2_lc_relative_covariance_01.dat',
                 name=None, version=None):
        self.name = name
        self.version = version
        self._model = {}
        self._parameters = np.array([1., 0., 0.])

        names_or_objs = {'M0': m0file, 'M1': m1file,
                         'LCRV00': lcrv00file, 'LCRV11': lcrv11file,
                         'LCRV01': lcrv01file, 'errscale': errscalefile,
                         'cdfile': cdfile, 'clfile': clfile}

        # Make filenames into full paths.
        if modeldir is not None:
            for k in names_or_objs:
                v = names_or_objs[k]
                if (v is not None and isinstance(v, str)):
                    names_or_objs[k] = os.path.join(modeldir, v)

        # model components are interpolated to 2nd order
        for key in ['M0', 'M1']:
            phase, wave, values = read_griddata_ascii(names_or_objs[key])
            values *= self._SCALE_FACTOR
            self._model[key] = BicubicInterpolator(phase, wave, values)

            # The "native" phases and wavelengths of the model are those
            # of the first model component.
            if key == 'M0':
                self._phase = phase
                self._wave = wave

        # model covariance is interpolated to 1st order
        for key in ['LCRV00', 'LCRV11', 'LCRV01', 'errscale']:
            phase, wave, values = read_griddata_ascii(names_or_objs[key])
            self._model[key] = BicubicInterpolator(phase, wave, values)

        # Set the colorlaw based on the "color correction" file.
        self._set_colorlaw_from_file(names_or_objs['clfile'])

        # Set the color dispersion from "color_dispersion" file
        w, val = np.loadtxt(names_or_objs['cdfile'], unpack=True)
        self._colordisp = Spline1d(w, val,  k=1)  # linear interp.

    def _flux(self, phase, wave):
        m0 = self._model['M0'](phase, wave)
        m1 = self._model['M1'](phase, wave)
        return (self._parameters[0] * (m0 + self._parameters[1] * m1) *
                10. ** (-0.4 * self._colorlaw(wave) * self._parameters[2]))

    def _bandflux_rvar_single(self, band, phase):
        """Model relative variance for a single bandpass."""

        # Raise an exception if bandpass is out of model range.
        if (band.minwave() < self._wave[0] or band.maxwave() > self._wave[-1]):
            raise ValueError('bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                             'outside spectral range [{3:.6g}, .., {4:.6g}]'
                             .format(band.name, band.wave[0], band.wave[-1],
                                     self._wave[0], self._wave[-1]))

        x1 = self._parameters[1]

        # integrate m0 and m1 components
        wave, dwave = integration_grid(band.minwave(), band.maxwave(),
                                       MODEL_BANDFLUX_SPACING)
        trans = band(wave)
        m0 = self._model['M0'](phase, wave)
        m1 = self._model['M1'](phase, wave)
        tmp = trans * wave
        f0 = np.sum(m0 * tmp, axis=1) * dwave / HC_ERG_AA
        m1int = np.sum(m1 * tmp, axis=1) * dwave / HC_ERG_AA
        ftot = f0 + x1 * m1int

        # In the following, the "[:,0]" reduces from a 2-d array of shape
        # (nphase, 1) to a 1-d array.
        lcrv00 = self._model['LCRV00'](phase, band.wave_eff)[:, 0]
        lcrv11 = self._model['LCRV11'](phase, band.wave_eff)[:, 0]
        lcrv01 = self._model['LCRV01'](phase, band.wave_eff)[:, 0]
        scale = self._model['errscale'](phase, band.wave_eff)[:, 0]

        v = lcrv00 + 2.0 * x1 * lcrv01 + x1 * x1 * lcrv11

        # v is supposed to be variance but can go negative
        # due to interpolation.  Correct negative values to some small
        # number. (at present, use prescription of snfit : set
        # negatives to 0.0001)
        v[v < 0.0] = 0.0001

        # avoid warnings due to evaluating 0. / 0. in f0 / ftot
        with np.errstate(invalid='ignore'):
            result = v * (f0 / ftot)**2 * scale**2

        # treat cases where ftot is negative the same as snfit
        result[ftot <= 0.0] = 10000.

        return result

    def bandflux_rcov(self, band, phase):
        """Return the *relative* model covariance (or "model error") on
        synthetic photometry generated from the model in the given restframe
        band(s).

        This model covariance represents the scatter of real SNe about
        the model.  The covariance matrix has two components. The
        first component is diagonal (pure variance) and depends on the
        phase :math:`t` and bandpass central wavelength
        :math:`\\lambda_c` of each photometry point:

        .. math::

           (F_{0, \\mathrm{band}}(t) / F_{1, \\mathrm{band}}(t))^2
           S(t, \\lambda_c)^2
           (V_{00}(t, \\lambda_c) + 2 x_1 V_{01}(t, \\lambda_c) +
            x_1^2 V_{11}(t, \\lambda_c))

        where the 2-d functions :math:`S`, :math:`V_{00}`, :math:`V_{01}`,
        and :math:`V_{11}` are given by the files ``errscalefile``,
        ``lcrv00file``, ``lcrv01file``, and ``lcrv11file``
        respectively and :math:`F_0` and :math:`F_1` are given by

        .. math::

           F_{0, \\mathrm{band}}(t) = \\int_\\lambda M_0(t, \\lambda)
                                      T_\\mathrm{band}(\\lambda)
                                      \\frac{\\lambda}{hc} d\\lambda

        .. math::

           F_{1, \\mathrm{band}}(t) = \\int_\\lambda
                                      (M_0(t, \\lambda) + x_1 M_1(t, \\lambda))
                                      T_\\mathrm{band}(\\lambda)
                                      \\frac{\\lambda}{hc} d\\lambda

        As this first component can sometimes be negative due to
        interpolation, there is a floor applied wherein values less than zero
        are set to ``0.01**2``. This is to match the behavior of the
        original SALT2 code, snfit.

        The second component is block diagonal. It has
        constant covariance between all photometry points within a
        bandpass (regardless of phase), and no covariance between
        photometry points in different bandpasses:

        .. math::

           CD(\\lambda_c)^2

        where the 1-d function :math:`CD` is given by the file ``cdfile``.
        Adding these two components gives the *relative* covariance on model
        photometry.

        Parameters
        ----------
        band : `~numpy.ndarray` of `~sncosmo.Bandpass`
            Bandpasses of observations.
        phase : `~numpy.ndarray` (float)
            Phases of observations.


        Returns
        -------
        rcov : `~numpy.ndarray`
            Model relative covariance for given bandpasses and phases.
        """

        # construct covariance array with relative variance on diagonal
        diagonal = np.zeros(phase.shape, dtype=np.float64)
        for b in set(band):
            mask = band == b
            diagonal[mask] = self._bandflux_rvar_single(b, phase[mask])
        result = np.diagflat(diagonal)

        # add kcorr errors
        for b in set(band):
            mask1d = band == b
            mask2d = mask1d * mask1d[:, None]  # mask for result array
            kcorrerr = self._colordisp(b.wave_eff)
            result[mask2d] += kcorrerr**2

        return result

    def _set_colorlaw_from_file(self, name_or_obj):
        """Read color law file and set the internal colorlaw function."""

        # Read file
        if isinstance(name_or_obj, str):
            f = open(name_or_obj, 'r')
        else:
            f = name_or_obj
        words = f.read().split()
        f.close()

        # Get colorlaw coeffecients.
        ncoeffs = int(words[0])
        colorlaw_coeffs = [float(word) for word in words[1: 1 + ncoeffs]]

        # If there are more than 1+ncoeffs words in the file, we expect them to
        # be of the form `keyword value`.
        version = None
        colorlaw_range = [3000., 7000.]
        for i in range(1+ncoeffs, len(words), 2):
            if words[i] == 'Salt2ExtinctionLaw.version':
                version = int(words[i+1])
            elif words[i] == 'Salt2ExtinctionLaw.min_lambda':
                colorlaw_range[0] = float(words[i+1])
            elif words[i] == 'Salt2ExtinctionLaw.max_lambda':
                colorlaw_range[1] = float(words[i+1])
            else:
                raise RuntimeError("Unexpected keyword: {}".format(words[i]))

        # Set extinction function to use.
        if version == 0:
            raise RuntimeError("Salt2ExtinctionLaw.version 0 not supported.")
        elif version == 1:
            self._colorlaw = SALT2ColorLaw(colorlaw_range, colorlaw_coeffs)
        else:
            raise RuntimeError('unrecognized Salt2ExtinctionLaw.version: ' +
                               version)

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
        """

        if wave is None:
            wave = self._wave
        else:
            wave = np.asarray(wave)
        if wave.ndim == 0:
            return self._colorlaw(np.ravel(wave))[0]
        else:
            return self._colorlaw(wave)


class SALT3Source(SALT2Source):
    """The SALT3 Type Ia supernova spectral timeseries model.
    Kenworthy et al., 2021, ApJ, submitted.  Model definitions
    are the same as SALT2 except for the errors, which are now
    given in flux space.  Unlike SALT2, no file is used for scaling
    the errors.

    The spectral flux density of this model is given by

    .. math::

       F(t, \\lambda) = x_0 (M_0(t, \\lambda) + x_1 M_1(t, \\lambda))
                       \\times 10^{-0.4 CL(\\lambda) c}

    where ``x0``, ``x1`` and ``c`` are the free parameters of the model,
    ``M_0``, ``M_1`` are the zeroth and first components of the model, and
    ``CL`` is the colorlaw, which gives the extinction in magnitudes for
    ``c=1``.

    Parameters
    ----------
    modeldir : str, optional
        Directory path containing model component files. Default is `None`,
        which means that no directory is prepended to filenames when
        determining their path.
    m0file, m1file, clfile : str or fileobj, optional
        Filenames of various model components. Defaults are:

        * m0file = 'salt2_template_0.dat' (2-d grid)
        * m1file = 'salt2_template_1.dat' (2-d grid)
        * clfile = 'salt2_color_correction.dat'

    lcrv00file, lcrv11file, lcrv01file, cdfile : str or fileobj
        (optional) Filenames of various model components for
        model covariance in synthetic photometry. See
        ``bandflux_rcov`` for details.  Defaults are:

        * lcrv00file = 'salt2_lc_relative_variance_0.dat' (2-d grid)
        * lcrv11file = 'salt2_lc_relative_variance_1.dat' (2-d grid)
        * lcrv01file = 'salt2_lc_relative_covariance_01.dat' (2-d grid)
        * cdfile = 'salt2_color_dispersion.dat' (1-d grid)

    Notes
    -----
    The "2-d grid" files have the format ``<phase> <wavelength>
    <value>`` on each line.

    The phase and wavelength values of the various components don't
    necessarily need to match. (In the most recent salt2 model data,
    they do not all match.) The phase and wavelength values of the
    first model component (in ``m0file``) are taken as the "native"
    sampling of the model, even though these values might require
    interpolation of the other model components.

    """

    _param_names = ['x0', 'x1', 'c']
    param_names_latex = ['x_0', 'x_1', 'c']
    _SCALE_FACTOR = 1e-12

    def __init__(self, modeldir=None,
                 m0file='salt3_template_0.dat',
                 m1file='salt3_template_1.dat',
                 clfile='salt3_color_correction.dat',
                 cdfile='salt3_color_dispersion.dat',
                 lcrv00file='salt3_lc_variance_0.dat',
                 lcrv11file='salt3_lc_variance_1.dat',
                 lcrv01file='salt3_lc_covariance_01.dat',
                 name=None, version=None):

        self.name = name
        self.version = version
        self._model = {}
        self._parameters = np.array([1., 0., 0.])

        names_or_objs = {'M0': m0file, 'M1': m1file,
                         'LCRV00': lcrv00file, 'LCRV11': lcrv11file,
                         'LCRV01': lcrv01file,
                         'cdfile': cdfile, 'clfile': clfile}

        # Make filenames into full paths.
        if modeldir is not None:
            for k in names_or_objs:
                v = names_or_objs[k]
                if (v is not None and isinstance(v, str)):
                    names_or_objs[k] = os.path.join(modeldir, v)

        # model components are interpolated to 2nd order
        for key in ['M0', 'M1']:
            phase, wave, values = read_griddata_ascii(names_or_objs[key])
            values *= self._SCALE_FACTOR
            self._model[key] = BicubicInterpolator(phase, wave, values)

            # The "native" phases and wavelengths of the model are those
            # of the first model component.
            if key == 'M0':
                self._phase = phase
                self._wave = wave

        # model covariance is interpolated to 1st order
        for key in ['LCRV00', 'LCRV11', 'LCRV01']:
            phase, wave, values = read_griddata_ascii(names_or_objs[key])
            values *= self._SCALE_FACTOR**2.
            self._model[key] = BicubicInterpolator(phase, wave, values)

        # Set the colorlaw based on the "color correction" file.
        self._set_colorlaw_from_file(names_or_objs['clfile'])

        # Set the color dispersion from "color_dispersion" file
        w, val = np.loadtxt(names_or_objs['cdfile'], unpack=True)
        self._colordisp = Spline1d(w, val,  k=1)  # linear interp.

    def _bandflux_rvar_single(self, band, phase):
        """Model relative variance for a single bandpass."""

        # Raise an exception if bandpass is out of model range.
        if (band.minwave() < self._wave[0] or band.maxwave() > self._wave[-1]):
            raise ValueError('bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                             'outside spectral range [{3:.6g}, .., {4:.6g}]'
                             .format(band.name, band.wave[0], band.wave[-1],
                                     self._wave[0], self._wave[-1]))

        x1 = self._parameters[1]

        # integrate m0 and m1 components
        wave, dwave = integration_grid(band.minwave(), band.maxwave(),
                                       MODEL_BANDFLUX_SPACING)
        trans = band(wave)
        m0 = self._model['M0'](phase, wave)
        m1 = self._model['M1'](phase, wave)
        tmp = trans * wave

        # evaluate avg M0 + x1*M1 across a bandpass
        f0 = np.sum(m0 * tmp, axis=1)/tmp.sum()
        m1int = np.sum(m1 * tmp, axis=1)/tmp.sum()
        ftot = f0 + x1 * m1int

        # In the following, the "[:,0]" reduces from a 2-d array of shape
        # (nphase, 1) to a 1-d array.
        lcrv00 = self._model['LCRV00'](phase, band.wave_eff)[:, 0]
        lcrv11 = self._model['LCRV11'](phase, band.wave_eff)[:, 0]
        lcrv01 = self._model['LCRV01'](phase, band.wave_eff)[:, 0]

        # variance in M0 + x1*M1 at the effective wavelength
        # of a bandpass
        v = lcrv00 + 2.0 * x1 * lcrv01 + x1 * x1 * lcrv11

        # v is supposed to be variance but can go negative
        # due to interpolation.  Correct negative values to some small
        # number. (at present, use prescription of snfit : set
        # negatives to 0.0001)
        v[v < 0.0] = 0.0001

        # avoid warnings due to evaluating 0. / 0. in f0 / ftot
        with np.errstate(invalid='ignore'):
            # turn M0+x1*M1 error into a relative error
            result = v/ftot**2.

        # treat cases where ftot is negative the same as snfit
        result[ftot <= 0.0] = 10000.
        return result


class MLCS2k2Source(Source):
    """A spectral time series model based on the MLCS2k2 model light curves,
    using the Hsiao template at each phase, mangled to match the model
    photometry.

    The spectral flux density of this model is given by

    .. math::

       F(t, \\lambda) = A \\times M(\\Delta, t, \\lambda)

    where _A_ is the amplitude and _Delta_ is the MLCS2k2 light curve shape
    parameter.

    .. note:: Requires scipy version 0.14 or higher.

    Parameters
    ----------
    fluxfile : str or obj
        Filename (or open file-like object) of a FITS file containing 3-d
        array of spectral flux density values for a grid of delta, phase
        and wavelength values.
    """

    _param_names = ['amplitude', 'delta']
    param_names_latex = ['A', '\\Delta']

    def __init__(self, fluxfile, name=None, version=None):

        # RegularGridInterpolator is only available in recent scipy
        # versions.
        try:
            from scipy.interpolate import RegularGridInterpolator
        except ImportError:
            import scipy  # to get scipy version
            raise ImportError("scipy version 0.14 or greater required for "
                              "MLCS2k2Source. Installed version: " +
                              scipy.__version__)

        self.name = name
        self.version = version
        self._parameters = np.array([1., 0.])

        delta, phase, wave, values = read_griddata_fits(fluxfile)

        self._phase = phase
        self._wave = wave
        self._delta = delta
        self._3d_model_flux = RegularGridInterpolator((delta, phase, wave),
                                                      values,
                                                      bounds_error=False,
                                                      fill_value=0.)

    def _flux(self, phase, wave):
        # "outer cartesian product" code from fast cartesian_product2 from
        # http://stackoverflow.com/questions/11144513/numpy-cartesian-product-
        #     of-x-and-y-array-points-into-single-array-of-2d-points
        arrays = [[self.parameters[1]], phase, wave]
        lp = len(phase)
        lw = len(wave)
        arr = np.empty((1, lp, lw, 3), dtype=np.float64)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        points = arr.reshape((-1, 3))
        return (self._parameters[0] *
                self._3d_model_flux(points).reshape(lp, lw))


class SNEMOSource(Source):
    """The SNEMO Type Ia supernova spectral timeseries model

    The spectral flux density of this model is given by

    .. math::
       F(t, \\lambda) = c_0(e_0(t, \\lambda) +
                           \\sum_{i=1}^{n} c_i e_i(t, \\lambda))
                           \\times FM07(\\lambda, A_s)

    where ``c_0``, ``c_i``, and ``A_s`` are the free parameters of the model.

    Parameters
    ----------
    fluxfile : str or obj, optional
        Filename of an ascii file containing 2-d
        array of spectral flux density values for a grid of phase
        and wavelength values. Assuming columns ``phase``, ``wavelength``,
        ``e_0``, ``e_1``, ``e_2``...
    """
    def __init__(self, fluxfile, name=None, version=None):
        self.name = name
        self.version = version

        phase, wave, values = read_multivector_griddata_ascii(fluxfile)
        n_vector = values.shape[0]

        self._parameters = np.zeros(n_vector+1)
        self._parameters[0] = 1

        _param_names = ['c0', 'As', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
                        'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14']
        param_names_latex = ['c_0', 'A_s', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5',
                             'c_6', 'c_7', 'c_8', 'c_9', 'c_{10}', 'c_{11}',
                             'c_{12}', 'c_{13}', 'c_{14}']
        self._param_names = _param_names[:n_vector + 1]
        self.param_names_latex = param_names_latex[:n_vector + 1]

        self._phase = phase
        self._wave = wave

        self._model_fluxes = np.array([Spline2d(phase, wave,
                                                v, kx=2, ky=2)
                                       for v in values])

    def _flux(self, phase, wave):
        c_0 = self._parameters[0]
        A_s = self._parameters[1]
        color = extinction.fm07(wave * u.angstrom, A_s)
        model_fluxes = np.array([mf(phase, wave) for mf
                                 in self._model_fluxes])

        model_ev = c_0 * (model_fluxes[0] +
                          (self._parameters[2:, None, None] *
                           model_fluxes[1:]).sum(axis=0))

        model_c = 10**(-0.4 * color)

        return model_ev * model_c


class Model(_ModelBase):
    """An observer-frame model, composed of a Source and zero or more effects.

    Parameters
    ----------
    source : `~sncosmo.Source` or str
        The model for the spectral evolution of the source. If a string
        is given, it is used to retrieve a `~sncosmo.Source` from
        the registry.
    effects : list of `~sncosmo.PropagationEffect`
        List of `~sncosmo.PropagationEffect` instances to add.
    effect_names : list of str
        Names of effects (same length as `effects`). The names are used
        to label the parameters.
    effect_frames : list of str
        The frame that each effect is in (same length as `effects`).
        Must be one of {'rest', 'obs'}.

    Notes
    -----
    The Source and PropagationEffects are copied upon instanciation.

    Examples
    --------
    >>> model = sncosmo.Model(source='hsiao')

    """

    def __init__(self, source, effects=None,
                 effect_names=None, effect_frames=None):
        # Set parameter names, initial values (inital values set to zero)
        self._param_names = ['z', 't0']
        self.param_names_latex = ['z', 't_0']
        self._parameters = np.zeros(2, dtype=float)

        # Set source and add source parameter names
        self._source = get_source(source, copy=True)
        self._param_names.extend(self._source.param_names)
        self.param_names_latex.extend(self._source.param_names_latex)

        # Add PropagationEffects
        self._effects = []
        self._effect_names = []
        self._effect_frames = []
        if (effects is not None or effect_names is not None or
                effect_frames is not None):
            try:
                same_length = (len(effects) == len(effect_names) and
                               len(effects) == len(effect_frames))
            except TypeError:
                raise TypeError('effects, effect_names, and effect_frames '
                                'should all be iterables.')
            if not same_length:
                raise ValueError('effects, effect_names and effect_frames '
                                 'must have matching lengths')

            for effect, name, frame in zip(effects, effect_names,
                                           effect_frames):
                self._add_effect_partial(effect, name, frame)

        # sync
        self._sync_parameter_arrays()
        self._update_description()

    def add_effect(self, effect, name, frame):
        """
        Add a PropagationEffect to the model.

        Parameters
        ----------
        effect : `~sncosmo.PropagationEffect`
            Propagation effect.
        name : str
            Name of the effect.
        frame : {'rest', 'obs', 'free'}
        """
        self._add_effect_partial(effect, name, frame)
        self._sync_parameter_arrays()
        self._update_description()

    @property
    def source(self):
        """The Source instance."""
        return self._source

    @property
    def effect_names(self):
        """Names of propagation effects (list of str)."""
        return self._effect_names

    @property
    def effects(self):
        """List of constituent propagation effects."""
        return self._effects

    def _add_effect_partial(self, effect, name, frame):
        """Like 'add effect', but don't sync parameter arrays"""

        if not isinstance(effect, PropagationEffect):
            raise TypeError('effect is not a PropagationEffect')
        if frame not in ['rest', 'obs', 'free']:
            raise ValueError("frame must be one of: {'rest', 'obs', 'free'}")
        self._effects.append(cp(effect))
        self._effect_names.append(name)
        self._effect_frames.append(frame)

        # for 'free' effects, add a redshift parameter
        if frame == 'free':
            self._param_names.append(name + 'z')
            self.param_names_latex.append('{\\rm ' + name + '}\\,z')

        # add all of this effect's parameters
        for param_name in effect.param_names:
            self._param_names.append(name + param_name)
            self.param_names_latex.append('{\\rm ' + name + '}\\,' +
                                          param_name)

    def _sync_parameter_arrays(self):
        """Synchronize parameter names and parameter arrays between
        the aggregated parameters and those of the individual source and
        effects.

        This is a bit tricksy, pay attention! After this, self._parameters
        holds all the model parameters. The source._parameters and
        effect._parameters arrays (for each effect) are changed to reference
        self._parameters. This works because ``B = A[start:stop]`` on
        numpy arrays makes ``B`` a reference to a block of memory in ``A``.
        We take advantage of this to make the model and it's components
        reference the same block of memory, so that updates to the model's
        parameters are automatically reflected in the components.

        This assumes that we are in a state where self._effects and
        self._effect_frames have been set, and self._parameters is an
        iterable.
        """

        # save a reference to old parameter values, in case there are
        # effect redshifts that have been set.
        old_parameters = self._parameters

        # Calculate total length of model's parameter array
        l = 2 + len(self._source._parameters)
        for effect, frame in zip(self._effects, self._effect_frames):
            l += (frame == 'free') + len(effect._parameters)

        # allocate new array (zeros so that new 'free' effects redshifts
        # initialize to 0)
        self._parameters = np.zeros(l, dtype=float)

        # copy old parameters: we do this to make sure we copy
        # non-default values of any parameters that the model alone
        # holds, such as z, t0 and effect redshifts.
        self._parameters[0:len(old_parameters)] = old_parameters

        # cross-reference source's parameters
        pos = 2
        l = len(self._source._parameters)
        self._parameters[pos:pos+l] = self._source._parameters  # copy
        self._source._parameters = self._parameters[pos:pos+l]  # reference
        pos += l

        # initialize a list of ints that keeps track of where the redshift
        # parameter of each effect is. Value is 0 if effect_frame is not 'free'
        self._effect_zindicies = []

        # for each effect, cross-reference the effect's parameters
        for i in range(len(self._effects)):
            effect = self._effects[i]

            # for 'free' effects, add a redshift parameter
            if self._effect_frames[i] == 'free':
                self._effect_zindicies.append(pos)
                pos += 1
            else:
                self._effect_zindicies.append(-1)

            # add all of this effect's parameters
            l = len(effect._parameters)
            self._parameters[pos:pos+l] = effect._parameters  # copy
            effect._parameters = self._parameters[pos:pos+l]  # reference
            pos += l

    def _update_description(self):
        # Make a name for myself. We have to watch out for None values here.
        # If all constituents are None, name is None. Otherwise, replace
        # None's with '?'
        names = [self._source.name] + self._effect_names
        if all([name is None for name in names]):
            self.description = None
        else:
            names = ['?' if name is None else name for name in names]
            self.description = '+'.join(names)

    def mintime(self):
        """Minimum observer-frame time at which the model is defined."""
        return (self._parameters[1] +
                (1. + self._parameters[0]) * self._source.minphase())

    def maxtime(self):
        """Maximum observer-frame time at which the model is defined."""
        return (self._parameters[1] +
                (1. + self._parameters[0]) * self._source.maxphase())

    def minwave(self):
        """Minimum observer-frame wavelength of the model."""
        source_shift = (1. + self._parameters[0])
        max_minwave = self._source.minwave() * source_shift
        for effect, frame, zindex in zip(self._effects, self._effect_frames,
                                         self._effect_zindicies):
            effect_minwave = effect.minwave()
            if frame == 'rest':
                effect_minwave *= source_shift
            elif frame == 'free':
                effect_minwave *= (1. + self._parameters[zindex])
            if effect_minwave > max_minwave:
                max_minwave = effect_minwave
        return max_minwave

    def maxwave(self):
        """Maximum observer-frame wavelength of the model."""
        source_shift = (1. + self._parameters[0])
        min_maxwave = self._source.maxwave() * source_shift
        for effect, frame, zindex in zip(self._effects, self._effect_frames,
                                         self._effect_zindicies):
            effect_maxwave = effect.maxwave()
            if frame == 'rest':
                effect_maxwave *= source_shift
            elif frame == 'free':
                effect_maxwave *= (1. + self._parameters[zindex])
            if effect_maxwave < min_maxwave:
                min_maxwave = effect_maxwave
        return min_maxwave

    def _baseflux(self, time, wave):
        """Array flux function."""
        a = 1. / (1. + self._parameters[0])
        phase = (time - self._parameters[1]) * a
        restwave = wave * a

        # Note that below we multiply by the scale factor to conserve
        # bolometric luminosity.
        f = a * self._source._flux(phase, restwave)

        return f

    def _flux(self, time, wave):
        """Array flux function."""

        a = 1. / (1. + self._parameters[0])
        obsphase = (time - self._parameters[1])
        restphase = obsphase * a
        restwave = wave * a

        # Note that below we multiply by the scale factor to conserve
        # bolometric luminosity.
        f = a * self._source._flux(restphase, restwave)

        # Pass the flux through the PropagationEffects.
        for effect, frame, zindex in zip(self._effects, self._effect_frames,
                                         self._effect_zindicies):
            if frame == 'obs':
                effect_wave = wave
                effect_phase = obsphase
            elif frame == 'rest':
                effect_wave = restwave
                effect_phase = restphase
            else:  # frame == 'free'
                effect_a = 1. / (1. + self._parameters[zindex])
                effect_wave = wave * effect_a
                effect_phase = obsphase * effect_a
            try:
                f = effect.propagate(effect_wave, f, phase=effect_phase)
            except TypeError:
                f = effect.propagate(effect_wave, f)
        return f

    def flux(self, time, wave):
        """The spectral flux density at the given time and wavelength values.

        Parameters
        ----------
        time : float or list_like
            Time(s) in days. If `None` (default), the times corresponding
            to the native phases of the model are used.
        wave : float or list_like
            Wavelength(s) in Angstroms. If `None` (default), the native
            wavelengths of the model are used.

        Returns
        -------
        flux : float or `~numpy.ndarray`
            Spectral flux density values in ergs / s / cm^2 / Angstrom.
        """

        time = np.asarray(time)
        wave = np.asarray(wave)

        # Check wavelength values
        if np.any(wave < self.minwave()) or np.any(wave > self.maxwave()):
            raise ValueError('requested wavelength value(s) outside '
                             'model range')

        # Get the flux
        try:
            f = self._flux(time, wave)
        except ValueError as e:
            _check_for_fitpack_error(e, time, 'time')
            _check_for_fitpack_error(e, wave, 'wave')
            raise e

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
        overlap = np.empty((len(band), len(z)), dtype=bool)
        shift = (1. + z)/(1+self._parameters[0])
        for i, b in enumerate(band):
            b = get_bandpass(b)
            overlap[i, :] = ((b.minwave() > self.minwave() * shift) &
                             (b.maxwave() < self.maxwave() * shift))
        if ndim == (0, 0):
            return overlap[0, 0]
        if ndim[1] == 0:
            return overlap[:, 0]
        return overlap

    def bandflux(self, band, time, zp=None, zpsys=None):
        """Flux through the given bandpass(es) at the given time(s).

        Default return value is flux in photons / s / cm^2. If zp and zpsys
        are given, flux(es) are scaled to the requested zeropoints.

        Parameters
        ----------
        band : str or list_like
            Name(s) of Bandpass(es) in registry.
        time : float or list_like
            Time(s) in days.
        zp : float or list_like, optional
            If given, zeropoint to scale flux to (must also supply ``zpsys``).
            If not given, flux is not scaled.
        zpsys : str or list_like, optional
            Name of a magnitude system in the registry, specifying the system
            that ``zp`` is in.

        Returns
        -------
        bandflux : float or `~numpy.ndarray`
            Flux in photons / s /cm^2, unless `zp` and `zpsys` are
            given, in which case flux is scaled so that it corresponds
            to the requested zeropoint. Return value is `float` if all
            input parameters are scalars, `~numpy.ndarray` otherwise.
        """

        try:
            return _bandflux(self, band, time, zp, zpsys)
        except ValueError as e:
            _check_for_fitpack_error(e, time, 'time')
            raise e

    def _bandflux_rcov(self, band, time):
        """Relative covariance in given bandpass and times.

        Parameters
        ----------
        band : str or list_like
            Name(s) of Bandpass(es) in registry.
        time : float or list_like
            Time(s) in days. Must be in ascending order.
        """

        a = 1. / (1. + self._parameters[0])

        # convert to 1-d arrays
        time, band = np.broadcast_arrays(time, band)
        ndim = time.ndim  # save input ndim for return val
        time = np.atleast_1d(time)
        band = np.atleast_1d(band)

        # Convert `band` to an array of rest-frame bands
        restband = np.empty(len(time), dtype='object')
        for b in set(band):
            mask = band == b
            b = get_bandpass(b)
            restband[mask] = b.shifted(a)

        phase = (time - self._parameters[1]) * a

        # Note that not all sources have this method. The idea
        # is that this will automatically fail if the method doesn't exist
        # for self._source.
        rcov = self._source.bandflux_rcov(restband, phase)

        if ndim == 0:
            return rcov[0, 0]
        return rcov

    def bandfluxcov(self, band, time, zp=None, zpsys=None):
        """Like bandflux(), but also returns model covariance on values.

        Parameters
        ----------
        band : `~sncosmo.bandpass` or str or list_like
            Bandpass(es) or name(s) of bandpass(es) in registry.
        time : float or list_like
            time(s) in days.
        zp : float or list_like, optional
            If given, zeropoint to scale flux to. if `none` (default) flux
            is not scaled.
        zpsys : `~sncosmo.magsystem` or str (or list_like), optional
            Determines the magnitude system of the requested zeropoint.
            cannot be `none` if `zp` is not `none`.

        Returns
        -------
        bandflux : float or `~numpy.ndarray`
            Model bandfluxes.
        cov : float or `~numpy.array`
            Covariance on ``bandflux``. If ``bandflux`` is an array, this
            will be a 2-d array.
        """

        f = self.bandflux(band, time, zp=zp, zpsys=zpsys)
        rcov = self._bandflux_rcov(band, time)

        if isinstance(f, np.ndarray):
            cov = f * rcov * f[:, np.newaxis]
        else:
            cov = f * rcov * f

        return f, cov

    def bandmag(self, band, magsys, time):
        """Magnitude at the given time(s) through the given
        bandpass(es), and for the given magnitude system(s).

        Parameters
        ----------
        band : str or list_like
            Name(s) of bandpass in registry.
        magsys : str or list_like
            Name(s) of `~sncosmo.MagSystem` in registry.
        time : float or list_like
            Observer-frame time(s) in days.

        Returns
        -------
        mag : float or `~numpy.ndarray`
            Magnitude for each item in time, band, magsys.
            The return value is a float if all parameters are not interables.
            The return value is an `~numpy.ndarray` if any are interable.
        """
        return _bandmag(self, band, magsys, time)

    def color(self, band1, band2, magsys, time):
        """band1 - band2 color at the given time(s) through the given pair of
        bandpasses, and for the given magnitude system.

        Parameters
        ----------
        band1 : str
            Name of first bandpass in registry.
        band2 : str
            Name of second bandpass in registry.
        magsys : str
            Name of `~sncosmo.MagSystem` in registry.
        time : float or list_like
            Observer-frame time(s) in days.

        Returns
        -------
        mag : float or `~numpy.ndarray`
            Color for each item in time, band, magsys.
            The return value is a float if all parameters are not iterables.
            The return value is an `~numpy.ndarray` if phase is iterable.
        """

        band1_isiterable = isiterable(band1) and not isinstance(band1, str)
        band2_isiterable = isiterable(band2) and not isinstance(band2, str)
        if band1_isiterable or band2_isiterable:
            raise TypeError("Band arguments must be scalars.")

        if (isiterable(magsys) and not isinstance(magsys, str)):
            raise TypeError("Magnitude system argument must be scalar.")

        return (self.bandmag(band1, magsys, time) -
                self.bandmag(band2, magsys, time))

    def source_peakmag(self, band, magsys, sampling=1.0):
        """Peak apparent magnitude of source in a rest-frame bandpass.

        Note that this is the peak magnitude of just the *source* component
        of the model, not including effects such as dust.

        Parameters
        ----------
        band : str or `~sncosmo.Bandpass`
            Bandpass or name of bandpass in registry.
        magsys : str or `~sncosmo.MagSystem`
            Magnitude system or name of magnitude system in registry.
        sampling : float, optional
            Sampling in rest-frame days used to find the peak of the light
            curve.

        Returns
        -------
        float
            Peak apparent magnitude of just the source component of the model.
        """

        return self._source.peakmag(band, magsys, sampling=sampling)

    def set_source_peakmag(self, m, band, magsys, sampling=1.0):
        """Set the amplitude of the source component of the model according to
        a peak apparent magnitude.

        Note that this is the peak magnitude of just the *source* component
        of the model, not including effects such as dust.

        Parameters
        ----------
        m : float
            Desired apparent magnitude.
        band : str or `~sncosmo.Bandpass`
            Bandpass or name of bandpass in registry.
        magsys : str or `~sncosmo.MagSystem`
            Magnitude system or name of magnitude system in registry.
        sampling : float, optional
            Sampling in rest-frame days used to find the peak of the light
            curve. Default is 1.0.
        """
        self._source.set_peakmag(m, band, magsys, sampling=sampling)

    def source_peakabsmag(self, band, magsys, sampling=1.0,
                          cosmo=cosmology.WMAP9):
        """Peak absolute magnitude of the source in rest-frame bandpass.

        Note that this is the peak absolute magnitude of just the *source*
        component of the model, not including effects such as dust.

        Parameters
        ----------
        band : str or `~sncosmo.Bandpass`
            Bandpass or name of bandpass in registry.
        magsys : str or `~sncosmo.MagSystem`
            Magnitude system or name of magnitude system in registry.
        sampling : float, optional
            Sampling in rest-frame days used to find the peak of the light
            curve. Default is 1.0.
        cosmo : astropy Cosmology, optional
            Instance of a cosmology from ``astropy.cosmology``, used to
            calculate distance modulus, given the model's redshift. Default
            is WMAP9.

        Returns
        -------
        float
            Peak absolute magnitude of just the source component of the model.
        """
        return (self._source.peakmag(band, magsys, sampling=sampling) -
                cosmo.distmod(self._parameters[0]).value)

    def set_source_peakabsmag(self, absmag, band, magsys, sampling=1.0,
                              cosmo=cosmology.WMAP9):
        """Set the amplitude of the source component of the model according to
        the desired absolute magnitude in the specified band.

        Parameters
        ----------
        absmag : float
            Desired absolute magnitude.
        band : str or `~sncosmo.Bandpass`
            Bandpass or name of bandpass in registry.
        magsys : str or `~sncosmo.MagSystem`
            Magnitude system or name of magnitude system in registry.
        sampling : float, optional
            Sampling in rest-frame days used to find the peak of the light
            curve. Default is 1.0.
        cosmo : astropy Cosmology, optional
            Instance of a cosmology from ``astropy.cosmology``, used to
            calculate distance modulus, given the model's redshift. Default
            is WMAP9.
        """

        if self._parameters[0] <= 0.:
            raise ValueError('absolute magnitude undefined when z<=0.')
        m = absmag + cosmo.distmod(self._parameters[0]).value
        self._source.set_peakmag(m, band, magsys, sampling=sampling)

    def _headsummary(self):
        head = "<{0:s} at 0x{1:x}>".format(self.__class__.__name__, id(self))
        s = 'source:\n' + self._source._headsummary()
        summaries = [head, s.replace('\n', '\n  ')]
        for effect, name, frame in zip(self._effects,
                                       self._effect_names,
                                       self._effect_frames):
            s = ('effect (name={0} frame={1}):\n{2}'
                 .format(repr(name), repr(frame), effect._headsummary()))
            summaries.append(s.replace('\n', '\n  '))
        return '\n'.join(summaries)

    def __copy__(self):
        new = Model(self._source,
                    effects=self._effects,
                    effect_names=self._effect_names,
                    effect_frames=self._effect_frames)
        new._parameters[:] = self._parameters
        return new

    def __deepcopy__(self, memo):
        return cp(self)


class PropagationEffect(_ModelBase):
    """Abstract base class for propagation effects.

    Derived classes must define _minwave (float), _maxwave (float).
    They may also define _minphase (float), and _maxphase (float).
    """

    __metaclass__ = abc.ABCMeta

    def minwave(self):
        return self._minwave

    def maxwave(self):
        return self._maxwave

    def minphase(self):
        try:
            return self._minphase
        except AttributeError:
            return np.nan

    def maxphase(self):
        try:
            return self._maxphase
        except AttributeError:
            return np.nan

    @abc.abstractmethod
    def propagate(self, wave, flux, phase=None):
        pass

    def _headsummary(self):
        summary = """\
        class           : {0}
        wavelength range: [{1:.6g}, {2:.6g}] Angstroms
        phase range     : [{3:.2g}, {4:.2g}]"""\
        .format(self.__class__.__name__,
                self._minwave, self._maxwave,
                self.minphase(), self.maxphase())
        return dedent(summary)


class CCM89Dust(PropagationEffect):
    """Cardelli, Clayton, Mathis (1989) extinction model dust."""
    _param_names = ['ebv', 'r_v']
    param_names_latex = ['E(B-V)', 'R_V']
    _minwave = 1000.
    _maxwave = 33333.33

    def __init__(self):
        self._parameters = np.array([0., 3.1])

    def propagate(self, wave, flux, phase=None):
        """Propagate the flux."""
        ebv, r_v = self._parameters
        return extinction.apply(extinction.ccm89(wave, ebv * r_v, r_v), flux)


class OD94Dust(PropagationEffect):
    """O'Donnell (1994) extinction model dust."""
    _param_names = ['ebv', 'r_v']
    param_names_latex = ['E(B-V)', 'R_V']
    _minwave = 909.09
    _maxwave = 33333.33

    def __init__(self):
        self._parameters = np.array([0., 3.1])

    def propagate(self, wave, flux, phase=None):
        """Propagate the flux."""
        ebv, r_v = self._parameters
        return extinction.apply(extinction.odonnell94(wave, ebv * r_v, r_v),
                                flux)


class F99Dust(PropagationEffect):
    """Fitzpatrick (1999) extinction model dust with fixed R_V."""
    _minwave = 909.09
    _maxwave = 60000.

    def __init__(self, r_v=3.1):
        self._param_names = ['ebv']
        self.param_names_latex = ['E(B-V)']
        self._parameters = np.array([0.])
        self._r_v = r_v
        self._f = extinction.Fitzpatrick99(r_v=r_v)

    def propagate(self, wave, flux, phase=None):
        """Propagate the flux."""
        ebv = self._parameters[0]
        return extinction.apply(self._f(wave, ebv * self._r_v), flux)


class G10(PropagationEffect):
    """Guy (2010) SNe Ia non-coherent scattering.
    
    Implementation is done following arxiv:1209.2482."""

    _param_names = ['L0', 'F0', 'F1', 'dL']
    param_names_latex = [r'\lambda_0', 'F_0', 'F_1', 'd_L']

    def __init__(self, SALTsource):
        """Initialize G10 class."""
        self._parameters = np.array([2157.3, 0.0, 1.08e-4, 800])
        self._colordisp = SALTsource._colordisp
        self._minwave = SALTsource.minwave()
        self._maxwave = SALTsource.maxwave()

    def compute_sigma_nodes(self):
        """Computes the sigma nodes."""
        L0, F0, F1, dL = self._parameters
        lam_nodes = np.arange(self._minwave, self._maxwave, dL)
        siglam_values = self._colordisp(lam_nodes) 
        
        siglam_values[lam_nodes < L0] *= 1 + (lam_nodes[lam_nodes < L0] - L0) * F0
        siglam_values[lam_nodes > L0] *= 1 + (lam_nodes[lam_nodes > L0] - L0) * F1
        siglam_values *= np.random.normal(size=len(lam_nodes))
        
        return lam_nodes, siglam_values

    def propagate(self, wave, flux):
        """Propagate the effect to the flux."""
        lam_nodes, siglam_values = self.compute_sigma_nodes()
        
        # Compute the sinus interpolation
        sup_bound = np.vstack([wave >= l for l in lam_nodes])
        idx_inf = np.sum(sup_bound, axis=0) - 1
        idx_inf[idx_inf==len(lam_nodes) - 1] = -2
        lam_node_inf = lam_nodes[idx_inf]
        lam_node_sup = lam_nodes[idx_inf + 1]
        smear_inf = siglam_values[idx_inf]
        smear_sup = siglam_values[idx_inf + 1]
        sin_interp = np.sin(np.pi * (wave - 0.5 * (lam_node_inf + lam_node_sup)) / (lam_node_sup - lam_node_inf))
        
        magscat = 0.5 * (smear_sup + smear_inf) + 0.5 * (smear_sup - smear_inf) * sin_interp
        
        return flux * 10**(-0.4 * magscat)