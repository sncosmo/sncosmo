# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Classes to represent spectral models of astronomical transients."""

import abc
import os
from copy import copy as cp
from textwrap import dedent
from math import ceil

import numpy as np
from scipy.interpolate import (InterpolatedUnivariateSpline as Spline1d,
                               RectBivariateSpline as Spline2d,
                               splmake, spleval)
from astropy.utils import OrderedDict as odict
from astropy import (cosmology,
                     units as u,
                     constants as const)

from .io import read_griddata_ascii
from . import registry
from .spectral import get_bandpass, get_magsystem, Bandpass
try:
    # Not guaranteed available at setup time
    from ._extinction import ccm89, od94, f99kknots, f99uv
except ImportError:
    if not _ASTROPY_SETUP_:
        raise

__all__ = ['get_source', 'Source', 'TimeSeriesSource', 'StretchSource',
           'SALT2Source', 'Model',
           'PropagationEffect', 'CCM89Dust', 'OD94Dust', 'F99Dust']

HC_ERG_AA = const.h.cgs.value * const.c.to(u.AA / u.s).value


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
        return cp(registry.retrieve(Source, name, version=version))


def _bandflux(model, band, time_or_phase, zp, zpsys):
    """Support function for bandflux in Source and Model.
    This is necessary to have outside because ``phase`` is used in Source
    and ``time`` is used in Model.
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
    bandflux = np.zeros(time_or_phase.shape, dtype=np.float)

    # Loop over unique bands.
    for b in set(band):
        mask = band == b
        b = get_bandpass(b)

        # Raise an exception if bandpass is out of model range.
        if (b.wave[0] < model.minwave() or b.wave[-1] > model.maxwave()):
            raise ValueError(
                'bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                'outside spectral range [{3:.6g}, .., {4:.6g}]'
                .format(b.name, b.wave[0], b.wave[-1],
                        model.minwave(), model.maxwave()))

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
        for key, val in param_dict.items():
            try:
                i = self._param_names.index(key)
            except ValueError:
                raise KeyError("Unknown parameter: " + repr(key))
            self._parameters[i] = val

    def get(self, name):
        try:
            i = self._param_names.index(name)
        except ValueError:
            raise KeyError("Model has no parameter " + repr(name))
        return self._parameters[i]

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


class Source(_ModelBase):
    """An abstract base class for transient models.

    A "transient model" in this case is the spectral time evolution
    of a source as a function of an arbitrary number of parameters.

    This is an abstract base class -- You can't create instances of
    this class. Instead, you must work with subclasses such as
    `TimeSeriesSource`. Subclasses must define (at minimum):

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
        phase : float or list_like
            Phase(s) in days.
        band : str or list_like
            Name(s) of bandpass in registry.
        magsys : str or list_like
            Name(s) of `~sncosmo.MagSystem` in registry.

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

        if isinstance(band_or_wave, (basestring, Bandpass)):
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

    def peakmag(self, band, magsys, sampling=1.):
        """Calculate peak apparent magnitude in rest-frame bandpass."""

        peakphase = self.peakphase(band, sampling=sampling)
        return self.bandmag(band, magsys, peakphase)

    def set_peakmag(self, m, band, magsys, sampling=1.):
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

    def summary(self):
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

       F(t, \lambda) = A \\times M(t, \lambda)

    where _M_ is the flux defined on a grid in phase and wavelength and _A_
    (amplitude) is the single free parameter of the model.

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

    def __init__(self, phase, wave, flux, name=None, version=None):
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._parameters = np.array([1.])
        self._model_flux = Spline2d(phase, wave, flux, kx=2, ky=2)

    def _flux(self, phase, wave):
        return self._parameters[0] * self._model_flux(phase, wave)


class StretchSource(Source):
    """A single-component spectral time series model, that "stretches" in
    time.

    The spectral flux density of this model is given by

    .. math::

       F(t, \lambda) = A \\times M(t / s, \lambda)

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
        self._model_flux = Spline2d(phase, wave, flux, kx=2, ky=2)

    def minphase(self):
        return self._parameters[1] * self._phase[0]

    def maxphase(self):
        return self._parameters[1] * self._phase[-1]

    def _flux(self, phase, wave):
        return (self._parameters[0] *
                self._model_flux(phase / self._parameters[1], wave))


class SALT2Source(Source):
    """The SALT2 Type Ia supernova spectral timeseries model.

    The spectral flux density of this model is given by

    .. math::

       F(t, \lambda) = x_0 (M_0(t, \lambda) + x_1 M_1(t, \lambda))
                       \\times CL(\lambda)^c

    where ``x0``, ``x1`` and ``c`` are the free parameters of the model.

    Parameters
    ----------
    modeldir : str, optional
        Directory path containing model component files. Default is `None`,
        which means that no directory is prepended to filenames when
        determining their path.
    m0file, m1file, clfile, cdfile, errscalefile,
    lcrv00file, lcrv11file, lcrv01file : str or fileobj, optional
        Filenames of various model components. Defaults are:

        * m0file = 'salt2_template_0.dat' (2-d grid)
        * m1file = 'salt2_template_1.dat' (2-d grid)
        * clfile = 'salt2_color_correction.dat'
        * errscalefile = 'salt2_lc_dispersion_scaling.dat' (2-d grid)
        * lcrv00file = 'salt2_lc_relative_variance_0.dat' (2-d grid)
        * lcrv11file = 'salt2_lc_relative_variance_1.dat' (2-d grid)
        * lcrv01file = 'salt2_lc_relative_covariance_01.dat' (2-d grid)
        * cdfile = 'salt2_color_dispersion.dat'

        The "2-d grid" files have the format ``<phase> <wavelength>
        <value>`` on each line. ``m0file``, ``m1file`` and ``clfile``
        together determine the model values. The ``cdfile`` and the
        model determine the "k-correction errors". ``errscalefile``,
        ``lcrv00file``, ``lcrv11file``, and ``lcrv01file`` determine the
        "error snake".

    Notes
    -----
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
                if (v is not None and isinstance(v, basestring)):
                    names_or_objs[k] = os.path.join(modeldir, v)

        # model components are interpolated to 2nd order
        for key in ['M0', 'M1']:
            phase, wave, values = read_griddata_ascii(names_or_objs[key])
            values *= self._SCALE_FACTOR
            self._model[key] = Spline2d(phase, wave, values, kx=2, ky=2)

            # The "native" phases and wavelengths of the model are those
            # of the first model component.
            if key == 'M0':
                self._phase = phase
                self._wave = wave

        # model covariance is interpolated to 1st order
        for key in ['LCRV00', 'LCRV11', 'LCRV01', 'errscale']:
            phase, wave, values = read_griddata_ascii(names_or_objs[key])
            self._model[key] = Spline2d(phase, wave, values, kx=1, ky=1)

        # Set the color dispersion from "color_dispersion" file
        w, val = np.loadtxt(names_or_objs['cdfile'], unpack=True)
        self._colordisp = Spline1d(w, val,  k=1)  # linear interp.

        # Set the colorlaw based on the "color correction" file.
        # Then interpolate it to the "native" wavelength grid,
        # for performance reasons.
        self._set_colorlaw_from_file(names_or_objs['clfile'])
        cl = self._colorlaw(self._wave)
        clbase = 10. ** (-0.4 * cl)
        self._model['clbase'] = Spline1d(self._wave, clbase, k=1)

    def _flux(self, phase, wave):
        m0 = self._model['M0'](phase, wave)
        m1 = self._model['M1'](phase, wave)
        return (self._parameters[0] * (m0 + self._parameters[1] * m1) *
                self._model['clbase'](wave)**self._parameters[2])

    def _errsnakesq(self, wave, phase):
        """Return the errorsnake squared (model variance) for the given
        rest-frame phases and rest-frame bandpass central wavelength.

        Note that this is "vectorized" on phase only. Wavelength
        must be a scalar.

        Parameters
        ----------
        wave : float
            Central wavelength of rest-frame bandpass.
        phase : 1-d `~numpy.ndarray`
            Restframe phases.

        Returns
        -------
        variance : 1-d `~numpy.ndarray`
        """

        x1 = self._parameters[1]

        # In the following, the "[:,0]" reduces from a 2-d array of shape
        # (nphase, 1) to a 1-d array.
        lcrv00 = self._model['LCRV00'](phase, wave)[:, 0]
        lcrv11 = self._model['LCRV11'](phase, wave)[:, 0]
        lcrv01 = self._model['LCRV01'](phase, wave)[:, 0]
        scale = self._model['errscale'](phase, wave)[:, 0]

        return scale*scale * (lcrv00 + 2*x1*lcrv01 + x1*x1*lcrv11)

    def _bandflux_rcov(self, band, phase):
        """Return the model relative covariance of integrated flux through
        the given restframe bands at the given phases

        band : 1-d `~numpy.ndarray` of `~sncosmo.Bandpass`
            Bandpasses of observations.
        phase : 1-d `~numpy.ndarray` (float)
            Phases of observations. Must be in ascending order.
        """

        x1 = self._parameters[1]

        # initialize integral arrays
        f0 = np.zeros(phase.shape, dtype=np.float)
        m1int = np.zeros(phase.shape, dtype=np.float)
        cwave = np.zeros(phase.shape, dtype=np.float)  # central wavelengths
        errsnakesq = np.empty(phase.shape, dtype=np.float)

        # Loop over unique bands
        for b in set(band):
            mask = band == b
            cwave[mask] = b.wave_eff

            # Raise an exception if bandpass is out of model range.
            if (b.wave[0] < self._wave[0] or b.wave[-1] > self._wave[-1]):
                raise ValueError(
                    'bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                    'outside spectral range [{3:.6g}, .., {4:.6g}]'
                    .format(b.name, b.wave[0], b.wave[-1],
                            self._wave[0], self._wave[-1]))

            m0 = self._model['M0'](phase[mask], b.wave)
            m1 = self._model['M1'](phase[mask], b.wave)

            tmp = b.trans * b.wave * b.dwave
            f0[mask] = np.sum(m0 * tmp, axis=1) / HC_ERG_AA
            m1int[mask] = np.sum(m1 * tmp, axis=1) / HC_ERG_AA

            errsnakesq[mask] = self._errsnakesq(b.wave_eff, phase[mask])

        # 2-d bool array of shape (len(band), len(band)):
        # true only where bands are same
        mask = cwave == cwave[:, np.newaxis]

        colorvar = self._colordisp(cwave)**2  # 1-d array
        colorcov = mask * colorvar  # 2-d * 1-d = 2-d

        # errorsnakesq is supposed to be variance but can go negative
        # due to interpolation.  Correct negative values to some small
        # number. (at present, use prescription of snfit : set
        # negatives to 0.0001)
        errsnakesq[errsnakesq < 0.] = 0.01*0.01

        f1 = f0 + x1 * m1int
        return colorcov + np.diagflat((f0 / f1)**2 * errsnakesq)

    def bandflux_rcov(self, band, phase):
        """Return the model relative covariance of integrated flux through
        the given restframe bands at the given phases

        band : `~numpy.ndarray` of `~sncosmo.Bandpass`
            Bandpasses of observations.
        phase : `~numpy.ndarray` (float)
            Phases of observations.
        """

        try:
            return _bandflux_rcov(self, band, phase)
        except ValueError as e:
            _check_for_fitpack_error(e, phase, 'phase')
            raise e

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
            raise Exception('unrecognized Salt2ExtinctionLaw.version: ' +
                            version)

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
        flux for arbitrary wavelengths.
        """
        if wave is None:
            wave = self._wave
        else:
            wave = np.asarray(wave)
        if wave.ndim == 0:
            return self._colorlaw(np.ravel(wave))[0]
        else:
            return self._colorlaw(wave)


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
    >>> model = sncosmo.Model(source='hsiao')  # doctest: +SKIP

    """

    def __init__(self, source, effects=None,
                 effect_names=None, effect_frames=None):
        self._param_names = ['z', 't0']
        self.param_names_latex = ['z', 't_0']
        self._parameters = np.array([0., 0.])
        self._source = get_source(source, copy=True)
        self.description = None
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
                raise TypeError('effects, effect_names, and effect_values '
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
            raise TypeError('effect is not a PropagationEffect')
        if frame not in ['rest', 'obs']:
            raise ValueError("frame must be one of: {'rest', 'obs'}")
        self._effects.append(cp(effect))
        self._effect_names.append(name)
        self._effect_frames.append(frame)
        self._synchronize_parameters()

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
            for name in effect.param_names_latex:
                self.param_names_latex.append('{\\rm ' + effect_name + '}\,' +
                                              name)

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
        shift = (1. + self._parameters[0])
        max_minwave = self._source.minwave() * shift
        for effect, frame in zip(self._effects, self._effect_frames):
            effect_minwave = effect.minwave()
            if frame == 'rest':
                effect_minwave *= shift
            if effect_minwave > max_minwave:
                max_minwave = effect_minwave
        return max_minwave

    def maxwave(self):
        """Maximum observer-frame wavelength of the model."""
        shift = (1. + self._parameters[0])
        min_maxwave = self._source.maxwave() * shift
        for effect, frame in zip(self._effects, self._effect_frames):
            effect_maxwave = effect.maxwave()
            if frame == 'rest':
                effect_maxwave *= shift
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
        overlap = np.empty((len(band), len(z)), dtype=np.bool)
        for i, b in enumerate(band):
            b = get_bandpass(b)
            overlap[i, :] = ((b.wave[0] > self._source.minwave() * (1. + z)) &
                             (b.wave[-1] < self._source.maxwave() * (1. + z)))
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
            restband[mask] = Bandpass(a*b.wave, b.trans)

        phase = (time - self._parameters[1]) * a

        # Note that not all sources have this method. The idea
        # is that this will automatically fail if the method doesn't exist
        # for self._source.
        rcov = self._source._bandflux_rcov(restband, phase)

        if ndim == 0:
            return rcov[0, 0]
        return rcov

    def bandfluxcov(self, band, time, zp=None, zpsys=None):
        """Like bandflux, but also returns model covariance on values.

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
        time : float or list_like
            Observer-frame time(s) in days.
        band : str or list_like
            Name(s) of bandpass in registry.
        magsys : str or list_like
            Name(s) of `~sncosmo.MagSystem` in registry.

        Returns
        -------
        mag : float or `~numpy.ndarray`
            Magnitude for each item in time, band, magsys.
            The return value is a float if all parameters are not interables.
            The return value is an `~numpy.ndarray` if any are interable.
        """
        return _bandmag(self, band, magsys, time)

    def source_peakabsmag(self, band, magsys, cosmo=cosmology.WMAP9):
        return (self._source.peakmag(band, magsys) -
                cosmo.distmod(self._parameters[0]).value)

    def set_source_peakabsmag(self, absmag, band, magsys,
                              cosmo=cosmology.WMAP9):
        """Set the amplitude of the source component of the model according to
        the desired absolute magnitude(s) in the specified band(s).

        Parameters
        ----------
        absmag : float
            Desired absolute magnitude.
        band : str or list_like
            Name(s) of bandpass in registry.
        magsys : str or list_like
            Name(s) of `~sncosmo.MagSystem` in registry.
        """

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
            s = ('effect (name={0} frame={1}):\n{2}'
                 .format(repr(name), repr(frame), effect.summary()))
            summaries.append(s.replace('\n', '\n  '))
        return '\n'.join(summaries)

    def __copy__(self):
        new = Model(self._source,
                    effects=self._effects,
                    effect_names=self._effect_names,
                    effect_frames=self._effect_frames)
        new._parameters[0:2] = self._parameters[0:2]
        return new


class PropagationEffect(_ModelBase):
    """Abstract base class for propagation effects.

    Derived classes must define _minwave (float), _maxwave (float).
    """

    __metaclass__ = abc.ABCMeta

    def minwave(self):
        return self._minwave

    def maxwave(self):
        return self._maxwave

    @abc.abstractmethod
    def propagate(self, wave, flux):
        pass

    def summary(self):
        summary = """\
        class           : {0}
        wavelength range: [{1:.6g}, {2:.6g}] Angstroms"""\
        .format(self.__class__.__name__, self._minwave, self._maxwave)
        return dedent(summary)


class CCM89Dust(PropagationEffect):
    """Cardelli, Clayton, Mathis (1989) extinction model dust."""
    _param_names = ['ebv', 'r_v']
    param_names_latex = ['E(B-V)', 'R_V']
    _minwave = 909.09
    _maxwave = 33333.33

    def __init__(self):
        self._parameters = np.array([0., 3.1])

    def propagate(self, wave, flux):
        """Propagate the flux."""
        a_v = self._parameters[0] * self._parameters[1]
        trans = 10.**(-0.4 * ccm89(wave, a_v, self._parameters[1]))
        return trans * flux


class OD94Dust(PropagationEffect):
    """O'Donnell (1994) extinction model dust."""
    _param_names = ['ebv', 'r_v']
    param_names_latex = ['E(B-V)', 'R_V']
    _minwave = 909.09
    _maxwave = 33333.33

    def __init__(self):
        self._parameters = np.array([0., 3.1])

    def propagate(self, wave, flux):
        """Propagate the flux."""
        a_v = self._parameters[0] * self._parameters[1]
        trans = 10.**(-0.4 * od94(wave, a_v, self._parameters[1]))
        return trans * flux


class F99Dust(PropagationEffect):
    """Fitzpatrick (1999) extinction model dust with fixed R_V."""
    _minwave = 909.09
    _maxwave = 60000.
    _XKNOTS = 1.e4 / np.array([np.inf, 26500., 12200., 6000., 5470.,
                               4670., 4110., 2700., 2600.])

    def __init__(self, r_v=3.1):
        self._param_names = ['ebv']
        self.param_names_latex = ['E(B-V)']
        self._parameters = np.array([0.])
        self._r_v = r_v

        kknots = f99kknots(self._XKNOTS, r_v)
        self._spline = splmake(self._XKNOTS, kknots, order=3)

    def propagate(self, wave, flux):

        ext = np.empty(len(wave), dtype=np.float)

        # Analytic function in the UV.
        uvmask = wave < 2700.
        if np.any(uvmask):
            a_v = self._parameters[0] * self._r_v
            ext[uvmask] = f99uv(wave[uvmask], a_v, self._r_v)

        # Spline in the Optical/IR
        oirmask = ~uvmask
        if np.any(oirmask):
            k = spleval(self._spline, 1.e4 / wave[oirmask])
            ext[oirmask] = self._parameters[0] * (k + self._r_v)

        trans = 10.**(-0.4 * ext)
        return trans * flux
