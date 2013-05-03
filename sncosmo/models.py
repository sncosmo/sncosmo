# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Classes to represent spectral models of astronomical transients."""

import abc
import os
import math
import copy
from textwrap import dedent

import numpy as np
from scipy.interpolate import RectBivariateSpline as Spline2d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline1d

import astropy.constants as const
import astropy.units as u
from astropy.utils import OrderedDict
from astropy.utils.misc import isiterable
from astropy import cosmology

from .utils import GridData1d, GridData2d, read_griddata, extinction_ccm
from .spectral import Spectrum, Bandpass, MagSystem, get_bandpass, get_magsystem
from . import registry

__all__ = ['get_model', 'Model', 'TimeSeriesModel', 'StretchModel',
           'SALT2Model']

cosmology.set_current(cosmology.WMAP9)
HC_ERG_AA = const.h.cgs.value * const.c.to(u.AA / u.s).value

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
    
    # Call the one in the registry to create a copy and keep the registry
    # copy pristine.
    return registry.retrieve(Model, name, version=version)()


class Model(object):
    """An abstract base class for transient models.
    
    A "transient model" in this case is the spectral time evolution
    of a source as a function of an arbitrary number of parameters.

    This is an abstract base class -- You can't create instances of
    this class. Instead, you must work with subclasses such as
    `TimeSeriesModel`. Subclasses must define (at minimum) `__init__()` and
    the private method `_model_flux()` 
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, name=None, version=None):
        self._params = OrderedDict(
            [('fscale', 1.), ('m', None), ('mabs', None), ('t0', 0.),
             ('z', None)])
        self._refphase = 0.
        self._refband = get_bandpass('bessellb')
        self._refmagsys = get_magsystem('ab')
        self._refm = None
        self._cosmo = cosmology.get_current()
        self._distmod = None  # Distance modulus
        self.name = name
        self.version = version

    def set(self, **params):
        """Set the parameters of the model using keyword values.

        Parameters
        ----------
        fscale : float
            The internal model flux spectral density values will be multiplied
            by this factor. If specified, `mabs` and `m` become unknown and
            are set to `None`.
        m : float
            Apparent magnitude; alternative way to set `fscale`.
        mabs : float
            Absolute magnitude; alternative way to set `fscale`. Apparent
            magnitude `m` is set according to the luminosity distance.
            Only permitted if luminosity distance is known (if cosmology is
            not None and z is not None).
        t0 : float
            Time offset.
        z : float
            Redshift.

        Notes
        -----
        `fscale` vs `mabs` vs `m` are various ways to set `fscale`;
        only one may be used at a time.
        """

        recalculate_mag = False
        recalculate_refm = False

        # Are multiple flux scalings methods specified?
        if ('fscale' in params) + ('m' in params) + ('mabs' in params) > 1:
            raise ValueError("Only one of 'fscale', 'm', 'mabs' may be set.")
        
        # Did the redshift (and distance modulus) change?
        if 'z' in params and self._cosmo is not None:
            if params['z'] is None:
                self._distmod = None
                self._params['mabs'] = None
            else:
                self._distmod = self._cosmo.distmod(params['z'])
                if self._params['mabs'] is not None:
                    recalculate_mag = True

        # Set parameters and check if any affect the magnitude
        for key in params:
            if key not in self._params: continue
            if key not in ['fscale', 't0', 'z']:
                recalculate_mag = True
                if key not in ['m', 'mabs']:
                    recalculate_refm = True
            self._params[key] = params[key]

        # If the flux scale is set directly, absolute and apparent
        # magnitude become unknown, similarly for apparent mag
        if 'fscale' in params:
            self._params['m'] = None
            self._params['mabs'] = None
        elif 'm' in params:
            self._params['mabs'] = None

        if recalculate_refm:
            self._calc_refmag()

        # Update fscale if we need to:
        if recalculate_mag:
            if self._params['mabs'] is not None:
                self._set_fscale_from_mabs()
            elif self._params['m'] is not None:
                self._set_fscale_from_m()

    def _calc_refmag(self):
        fscale = self._params['fscale']
        self._params['fscale'] = 1.
        self._refm = self.bandmag(self._refband, self._refmagsys,
                                  self._refphase, modelframe=True)
        self._params['fscale'] = fscale

    def _set_fscale_from_m(self):
        """Determine the fscale that when applied to the model
        (with current parameters), will result in the desired absolute
        magnitude at the reference phase."""

        if self._refm is None:
            self._calc_refmag()
        self._params['fscale'] = 10.**(0.4 * (self._refm - self._params['m']))

    def _set_fscale_from_mabs(self):
        """Determine the fscale that when applied to the model
        (with current parameters), will result in the desired absolute
        magnitude at the reference phase."""

        if self._distmod is None:
            raise ValueError("cannot set absolute magnitude when distance "
                             "modulus is unknown (when either redshift or "
                             "cosmology is None)")
        self._params['m'] = self._params['mabs'] + self._distmod
        self._set_fscale_from_m()

    @abc.abstractmethod
    def _model_flux(self, phase, disp):
        """Return the model spectral flux density (without any scaling or
        redshifting."""
        pass

    @property
    def parnames(self):
        return self._params.keys()

    @property
    def params(self):
        """Dictionary of model parameters. (Read-only.) Use `set` method to
        set parameters."""
        return copy.copy(self._params)

    @property
    def cosmo(self):
        """Cosmology used to calculate luminosity distance."""
        return self._cosmo

    @cosmo.setter
    def cosmo(self, new_cosmology):
        if (new_cosmology is not None and 
            not isinstance(new_cosmology, cosmology.FLRW)):
            raise ValueError('cosmology must be None or '
                             'astropy.cosmology.FLRW instance.')
        self._cosmo = new_cosmology
        
        # If z is set, we need to change distmod and recalculate fscale.
        if self._params['z'] is not None:
            if self._cosmo is None:
                self._distmod = None
            else:
                dm = self._cosmo.distmod(self._params['z'])
                self._distmod = dm
            if self._params['mabs'] is not None:
                self._set_fscale_from_mabs()

    @property
    def refphase(self):
        """Phase at which absolute magnitude is evaluated."""
        return self._refphase

    @refphase.setter
    def refphase(self, new_refphase):
        self._refphase = new_refphase

    @property
    def refband(self):
        """Bandpass in absolute magnitude is evaluated."""
        return self._refband

    @refband.setter
    def refband(self, band):
        self._refphase = get_bandpass(band)

    @property
    def refmagsys(self):
        """Magnitude system in which absolute magnitude is evaluated."""
        return self._refmagsys

    @refmagsys.setter
    def refmagsys(self, magsystem):
        self._refmagsys = get_magsystem(magsystem)


    def times(self, modelframe=False):
        """Return native time sampling of the model.

        Parameters
        ----------
        modelframe : bool, optional
            If True, return rest-frame phases. Default is False.
        """

        if modelframe:
            return self._phase
        
        z = self._params['z']
        if z is None: z = 0.

        return self._params['t0'] + (1. + z) * self._phase

    def disp(self, modelframe=False):
        """Return native dispersion grid of the model.

        Parameters
        ----------
        modelframe : bool, optional
            If True, return rest-frame phases. Default is False, which
            corresponds to observer-frame phases (model phases
            multiplied by 1 + z).
        """
        if modelframe or self._params['z'] is None:
            return self._disp
        else:
            return self._disp * (1. + self._params['z'])

    def flux(self, time=None, disp=None, modelframe=False):
        """The model flux spectral density at the given dispersion values.

        Parameters
        ----------
        time : float or list_like, optional
            Time(s) in days. Times are observer frame unless `modelframe`
            is True, in which case times are assumed to be model phases.
            If `None` (default) the native phases of the model are used.
        disp : float or list_like, optional
            Model dispersion values in Angstroms. Interpreted to be in the
            observer frame unless `modelframe` is True. If `None` (default)
            the native dispersion of the model is used.
        modelframe : bool, optional
            If True, return fluxes without redshifting or time shifting.

        Returns
        -------
        flux : float or `~numpy.ndarray`
            Spectral flux density values in ergs / s / cm^2 / Angstrom.
        """

        z = max(0., self._params['z'])

        # If not model frame, convert time and dispersion to model frame
        if not modelframe:
            if time is not None:
                time = np.asarray(time)
                time = (time - self._params['t0']) / (1. + z)
            if disp is not None:
                disp = np.asarray(disp)
                disp /= (1. + z)

        # Determine flux scaling factor
        factor = self._params['fscale']
        if not modelframe:
            factor /= 1. + z

        return factor * self._model_flux(time, disp)

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
        if z is None: z = self._params['z']
        if z is None: z = 0
        z = np.asarray(z)
        ndim = (band.ndim, z.ndim)
        band = band.ravel()
        z = z.ravel()
        overlap = np.empty((len(band), len(z)), dtype=np.bool)
        for i, b in enumerate(band):
            b = get_bandpass(b)
            overlap[i, :] = ((b.disp[0] > self._disp[0] * (1. + z)) &
                             (b.disp[-1] < self._disp[-1] * (1. + z)))
        if ndim == (0, 0):
            return overlap[0, 0]
        if ndim[1] == 0:
            return overlap[:, 0]
        return overlap

    def bandflux(self, band, time=None, zp=None, zpmagsys=None,
                 modelframe=False, include_error=False):
        """Flux through the given bandpass(es) at the given time(s).

        Default return value is flux in photons / s / cm^2. If zp and zpmagsys
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
        zpmagsys : `~sncosmo.MagSystem` or str (or list_like), optional
            Determines the magnitude system of the requested zeropoint.
            Cannot be `None` if `zp` is not `None`.
        include_error : bool, optional
            Include flux errors in return value. Default is False.


        Returns
        -------
        bandflux : float or `~numpy.ndarray`
            Flux in photons / s /cm^2, unless `zp` and `zpmagsys` are
            given, in which case flux is scaled so that it corresponds
            to the requested zeropoint. Return value is `float` if all
            input parameters are scalars, `~numpy.ndarray` otherwise.
        bandfluxerr : float or `~numpy.ndarray`
        """

        z = max(0., self._params['z'])
        if modelframe: z = 0.

        # Convert time to model frame (phase)
        if not modelframe and time is not None:
            time = np.asarray(time)
            time = (time - self._params['t0']) / (1. + z)
        if time is None:
            time = self._phase

        # Determine flux scaling factor
        factor = self._params['fscale']
        if not modelframe:
            factor /= 1. + z

        # broadcast arrays
        if zp is None:
            time, band = np.broadcast_arrays(time, band)
        else:
            if zpmagsys is None:
                raise ValueError('zpmagsys must be given if zp is not None')
            time, band, zp, zpmagsys = \
                np.broadcast_arrays(time, band, zp, zpmagsys)
            zp = np.atleast_1d(zp)
            zpmagsys = np.atleast_1d(zpmagsys)

        # convert to 1d arrays
        ndim = time.ndim # save input ndim for return val
        time = np.atleast_1d(time)
        band = np.atleast_1d(band)

        # initialize output arrays
        bandflux = np.zeros(time.shape, dtype=np.float)
        if include_error:
            relerr = np.zeros(time.shape, dtype=np.float)

        # index of times that are in model range
        idx_validtime = (time >= self._phase[0]) & (time <= self._phase[-1])

        # loop over unique bands
        for b in set(band):
            idx = (band == b) & idx_validtime
            if not np.any(idx): continue
            b = get_bandpass(b)
            d = b.disp / (1. + z)  # bandpass dispersion in modelframe
            
            # make sure bandpass dispersion is in model range.
            if (d[0] < self._disp[0] or d[-1] > self._disp[-1]):
                raise ValueError(
                    'bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                    'outside model range [{3:.6g}, .., {4:.6g}]'
                    .format(b.name, b.disp[0], b.disp[-1], 
                            (1.+z) * self._disp[0], (1.+z) * self._disp[-1]))
            flux = self._model_flux(time[idx], d)
            tmp = b.trans * b.disp * b.ddisp
            fluxsum = np.sum(flux * tmp, axis=1) / HC_ERG_AA

            if zp is not None:
                zpnorm = 10. ** (0.4*zp[idx])
                bandzpmagsys = zpmagsys[idx]
                for ms in set(bandzpmagsys):
                    idx2 = bandzpmagsys == ms
                    ms = get_magsystem(ms)
                    zpnorm[idx2] = zpnorm[idx2] / ms.zpbandflux(b)
                fluxsum *= zpnorm

            bandflux[idx] = fluxsum

            if include_error:
                deff = b.disp_eff
                if not modelframe: deff /= (1.+z)
                relerr[idx] = self._model_bandfluxrelerr(time[idx], deff)[:, 0]

        # multiply all by overall factor determined above.
        bandflux *= factor

        # Return result.
        if include_error:
            if ndim == 0:
                return bandflux[0], bandflux[0] * relerr[0]
            return bandflux, bandflux * relerr
        if ndim == 0:
            return bandflux[0]
        return bandflux

        # OLD ERROR CALCULATION:
        #if error:
        #    bandfluxerr[i] = \
        #        np.sqrt(np.sum((fluxerr[ti, idx]*d*dd)**2 * t)) / HC_ERG_AA

    def bandmag(self, band, magsys, time=None, modelframe=False):
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

        bandflux = self.bandflux(band, time, modelframe=modelframe)
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

    def set_bandflux_relative_error(self, phase, disp, relative_error):
        """Set the relative error to be applied to the bandflux.

        Parameters
        ----------
        phase : `~numpy.ndarray` (1d)
            Phases in days.
        disp : `~numpy.ndarray` (1d)
            Wavelength values.
        relative_error : `~numpy.ndarray` (2d)
            Fractional error on bandflux. Must have shape ``(len(phases),
            len(disp))``
        """
        self._model_bandfluxrelerr = \
            GridData2d(phase, disp, relative_error)

    def _phase_of_max_flux(self, band):
        """Determine phase of maximum flux in the given band, by fitting
        a parabola to phase of maximum flux and the surrounding two
        phases."""

        fluxes = self.bandflux(band)
        i = np.argmax(fluxes)
        x = self._phase[i-1: i+2]
        y = fluxes[i-1: i+2]
        A = np.hstack([x.reshape(3,1)**2, x.reshape(3,1), np.ones((3,1))])
        a, b, c = np.linalg.solve(A, y)
        return -b / (2 * a)


    def __call__(self, **params):
        """Return a shallow copy of the model with parameters set.
        See `set` method for details on parameters.
        
        Returns
        -------
        model : `sncosmo.Model`
            Model instance with parameters set as requested.

        Examples
        --------
        
        >>> model = sncosmo.get_model('salt2')
        >>> sn = model(c=0.1, x1=0.5, mabs=(-19.3, 'bessellb', 'ab'),
        ...            z=0.68)
        >>> sn.magnitude(0., 'desg', 'ab')
        25.577096820065883
        >>> sn = model(c=0., x1=0., mabs=(-19.3, 'bessellb', 'ab'),
        ...            z=0.5)
        >>> sn.magnitude(0., 'desg', 'ab')
        23.73028907970092
        """

        model = copy.copy(self)
        model._params = copy.copy(self._params)
        model.set(**params)
        return model

    def __repr__(self):
        name = ''
        version = ''
        if self.name is not None:
            name = ' {0!r:s}'.format(self.name)
        if self.version is not None:
            version = ' version={0!r:s}'.format(self.version)
        return "<{0:s}{1:s}{2:s} at 0x{3:x}>".format(
            self.__class__.__name__, name, version, id(self))


    def __str__(self):
        dmstr = '--'
        ldstr = '--'
        if self._distmod is not None:
            dmstr = '{:.6g}'.format(self._distmod)
            ld =  10.**((self._distmod - 25.) / 5.)
            ldstr = '{:.6g} Mpc'.format(ld)
        result = """\
        Model class: {}
        Model name: {}
        Model version: {}
        Model phases: [{:.6g}, .., {:.6g}] days ({:d} points)
        Model dispersion: [{:.6g}, .., {:.6g}] Angstroms ({:d} points) 
        Reference phase: {} days
        Cosmology: {}
        Current Parameters:
        """.format(
            self.__class__.__name__,
            self.name, self.version,
            self._phase[0], self._phase[-1], len(self._phase),
            self._disp[0], self._disp[-1],
            len(self._disp),
            self._refphase,
            self._cosmo)
        result = dedent(result)

        parameter_lines = []
        for key, val in self._params.iteritems():
            line = '    {} = {}'.format(key, val)
            if key in ['m', 'mabs']:
                line += ' [{}, {}]'.format(self._refband.name,
                                           self._refmagsys.name)
            elif key == 'z' and val is not None:
                line += ('[dist. mod. = {}, lum. dist. = {}]'
                         .format(dmstr, ldstr))
            parameter_lines.append(line)
        return result + '\n'.join(parameter_lines)


class TimeSeriesModel(Model):
    """A single-component spectral time series model.

    Parameters
    ----------
    phase : `~numpy.ndarray`
        Phases in days.
    disp : `~numpy.ndarray`
        Wavelengths in Angstroms.
    flux : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape `(num_phases, num_disp)`.
    extinction_func : function, optional
        A function that accepts an array of wavelengths in Angstroms
        and returns an array representing the ratio of total extinction
        (in magnitudes) to the parameter `c` at each wavelength. Default
        is `sncosmo.utils.extinction_ccm`.
    extinction_kwargs : dict
        A dictionary of keyword arguments to pass to `extinction_func`.
        Default is `dict(ebv=1.)`.
    name : str, optional
        Name of the model. Default is `None`.
    version : str, optional
        Version of the model. Default is `None`.
    """

    def __init__(self, phase, disp, flux,
                 extinction_func=extinction_ccm,
                 extinction_kwargs=dict(ebv=1.),
                 name=None, version=None):

        super(TimeSeriesModel, self).__init__(name, version)
        self._params['c'] = None

        self._phase = phase
        self._disp = disp
        self._flux = Spline2d(phase, disp, flux, kx=2, ky=2)
        self._model_bandfluxrelerr = Spline2d(np.array([phase[0], phase[-1]]),
                                              np.array([disp[0], disp[-1]]),
                                              np.zeros((2,2), dtype=np.float),
                                              kx=1, ky=1)
            
        self.set_extinction_func(extinction_func, extinction_kwargs)

        self._refphase = self._phase_of_max_flux(self._refband)

        self._precomputed = None

    def _model_flux(self, phase=None, disp=None):
        """Return the model flux density (without any scaling or redshifting).
        """
        if phase is None:
            phase = self._phase
        if disp is None:
            disp = self._disp
        if self._params['c'] is None:
            return self._flux(phase, disp)
        
        dust_trans = self._dust_trans_base(disp) ** self._params['c']
        return dust_trans * self._flux(phase, disp)

    def set_extinction_func(self, func, extra_params):
        """Set the extinction ratio function to use in the model.

        Parameters
        ----------
        func : function
            A function that accepts an array of wavelengths in Angstroms
            and returns the ratio of total extinction
            (in magnitudes) to the parameter `c` at each wavelength.
        extra_params : dict
            A dictionary of keyword arguments to pass to the function.
            Default is `None`.
        """
        
        # set these parameters for info
        self._extinction_func = func
        self._extinction_kwargs = extra_params

        if extra_params is None:
            ext_ratio = func(self._disp)
        else:
            ext_ratio = func(self._disp, **extra_params)

        # calculate extinction base values, so that
        # self._dust_trans_base ** c gives the dust transmission.
        dust_trans_base =  10. ** (-0.4 * ext_ratio)
        self._dust_trans_base = Spline1d(self._disp, dust_trans_base, k=2)

    def precompute(self, bands, z, c, verbose=False):
        """Precompute bandfluxes for given redshifts and color values."""
        
        self._precomputed = {'parnames': ['z', 'c'],
                             'parvals': [z, c],
                             'f': {},
                             'fe': {}}

        # save current parameters, and set fscale to 1.
        saved_params = copy.copy(self._params)
        self.set(fscale=1.)

        # Parameter grid size
        parshape = [len(v) for v in self._precomputed['parvals']]
        oshape = parshape + [len(self._phase)]
        ngrid = 1
        for d in parshape:
            ngrid *= d
        ishape = [ngrid, len(self._phase)]

        for band in bands:
            band = get_bandpass(band)
            name = band.name
            if name is None:
                raise ValueError('Bandpass must have name for precomputation')

            self._precomputed['f'][name] = np.empty(ishape, dtype=np.float)
            self._precomputed['fe'][name] = np.empty(ishape, dtype=np.float)

            for i, v in enumerate(product(*self._precomputed['parvals'])):
                model.set(**dict(zip(self._precomputed['parnames'], v)))
                f, fe = self.bandflux(band, include_error=True)
                self._precomputed['f'][name][i, :] = f
                self._precomputed['fe'][name][i, :] = fe

class StretchModel(TimeSeriesModel):
    """A single-component spectral time series model, that "stretches".

    Parameters
    ----------
    phase : `~numpy.ndarray`
        Phases in days.
    disp : `~numpy.ndarray`
        Wavelengths in Angstroms.
    flux : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape `(num_phases, num_disp)`.

    Other Parameters
    ----------------
    flux_error : `~numpy.ndarray`, optional
        Model error on `flux`. Must have same shape as `flux`.
        Default is `None`.
    extinction_func : function, optional
        A function that accepts an array of wavelengths in Angstroms
        and returns an array representing the ratio of total extinction
        (in magnitudes) to the parameter `c` at each wavelength. Default
        is `sncosmo.utils.extinction_ccm`
    extinction_kwargs : dict
        A dictionary of keyword arguments to pass to `extinction_func`.
        Default is `dict(ebv=1.)`.
    name : str, optional
        Name of the model. Default is `None`.
    version : str, optional
        Version of the model. Default is `None`.
    """
    def __init__(self, phase, disp, flux,
                 flux_error=None, name=None, version=None,
                 extinction_func=extinction_ccm,
                 extinction_kwargs=dict(ebv=1.)):

        super(StretchModel, self).__init__(
            phase, disp, flux,
            flux_error=flux_error,
            name=name, version=version,
            extinction_func=extinction_func,
            extinction_kwargs=extinction_kwargs)
        
        self._params['s'] = None

    def _model_flux(self, phase=None, disp=None):
        """Return the model flux density (without any scaling or redshifting).
        """
        if phase is not None and self._params['s'] is not None:
            phase = phase / self._params['s']
        return super(StretchModel, self)._model_flux(phase, disp)

    def _set_fscale_from_mabs(self, mabs):
        """Need to override this so that the refphase is applied correctly.
        We want refphase to refer to the *unstretched* phase. Otherwise the
        phase of peak will shift with s, if the peak is not at phase = 0."""

        s = self._params['s']
        self._params['s'] = None # temporarily remove stretch
        super(StretchModel, self)._set_fscale_from_mabs()
        self._params['s'] = s # put it back.


    def times(self, modelframe=False):
        """Return native phases of the model.

        Parameters
        ----------
        modelframe : bool, optional
            If True, return rest-frame phases. Default is False, which
            corresponds to observer-frame phases (model phases
            multiplied by 1 + z).
        """

        s = self._params['s']
        z = self._params['z']

        if (modelframe or z is None): 
            if s is None:
                return self._phase
            z = 0.
        if s is None:
            s = 1.

        return s * (1. + z) * self._phase


class SALT2Model(Model):
    """The SALT2 Type Ia supernova spectral timeseries model.

    Parameters
    ----------
    modeldir : str, optional
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
        super(SALT2Model, self).__init__(name, version)
        self._params['c'] = 0.
        self._params['x1'] = 0.
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
            self._model[component] = GridData2d(phase, wavelength, values)

            # The "native" phases and wavelengths of the model are those
            # of the first model component.
            if component == 'M0':
                self._phase = phase
                self._disp = wavelength

        # add extinction component
        ext_ratio = self._extinction(self._disp)
        dust_trans_base = 10. ** (-0.4 * ext_ratio)
        self._model['ext'] = GridData1d(self._disp, dust_trans_base)

        # Set relative bandflux error
        self._model_bandfluxrelerr = \
            GridData2d(np.array([self._phase[0], self._phase[-1]]),
                       np.array([self._disp[0], self._disp[-1]]),
                       np.zeros((2,2), dtype=np.float))

        # set refphase
        self._refphase = self._phase_of_max_flux(self._refband)

    def _model_flux(self, phase=None, disp=None):
        """Return the model flux density (without any scaling or redshifting).
        """

        if phase is None:
            phase = self._phase
        if disp is None:
            disp = self._disp
        f0 = self._model['M0'](phase, disp)
        f1 = self._model['M1'](phase, disp)
        flux = f0 + self._params['x1'] * f1
        flux *= self._model['ext'](disp) ** self._params['c']
        return flux

    # No longer supporting spectral flux error, so we comment out the 
    # following method. This might be added back at a later date if
    # I gain a better understanding of how to use the SALT2 model errors.

    #def _model_fluxerr(self, phase=None, disp=None):
    #    """Return the model flux density error (without any scaling or
    #    redshifting. Return `None` if the error is undefined.
    #    """
    #    if phase is None:
    #        phase = self._phase
    #    if disp is None:
    #        disp = self._disp
    #    v00 = self._model['V00'](phase, disp)
    #    v11 = self._model['V11'](phase, disp)
    #    v01 = self._model['V01'](phase, disp)

        #TODO apply x0 correctly
    #    x1 = self._params['x1']
    #    sigma = np.sqrt(v00 + x1**2 * v11 + 2 * x1 * v01)
    #    sigma *= self._model['ext'](disp) ** self._params['c']
        ### sigma *= 1e-12   #- Magic constant from SALT2 code
        
    #    if 'errscale' in self._model:
    #        sigma *= self._model['errscale'](phase, disp)
        
        # Hack adjustment to error (from SALT2 code)
        #TODO: figure out a way to do this.
        #if phase < -7. or phase > 12.:
        #    idx = (disp < 3400.)
        #    sigma[idx] *= 1000.
        
    #   return sigma


    def _extinction(self, wavelengths, params=[-0.336646, 0.0484495]):
        """Return the extinction as a function of wavelength, for c=1.

        Notes
        -----
        From SALT2 code comments:

            ext = exp(color * constant * 
                      (l + params(0)*l^2 + params(1)*l^3 + ... ) /
                      (1 + params(0) + params(1) + ... ) )
                = exp(color * constant *  numerator / denominator )
                = exp(color * expo_term ) 
        """

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

        return -numerator / denominator
