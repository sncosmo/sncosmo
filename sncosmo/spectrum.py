# Licensed under a 3-clause BSD style license - see LICENSE.rst

from copy import deepcopy
from collections import OrderedDict

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy import cosmology

from .bandpasses import get_bandpass

__all__ = ['Spectrum']


class Spectrum(object):
    """A spectrum, representing wavelength and spectral density values.

    Parameters
    ----------
    wave : list_like
        Wavelength values.
    flux : list_like
        Spectral flux density values.
    error : list_like, optional
        1 standard deviation uncertainty on flux density values.
    wave_unit : `~astropy.units.Unit`
        Units.
    unit : `~astropy.units.BaseUnit`
        For now, only units with flux density in energy (not photon counts).
    z : float, optional
        Redshift of spectrum (default is `None`)
    dist : float, optional
        Luminosity distance in Mpc, used to adjust flux upon redshifting.
        The default is ``None``.
    meta : OrderedDict, optional
        Metadata.
    """

    def __init__(self, wave, flux, error=None,
                 unit=(u.erg / u.s / u.cm**2 / u.AA), wave_unit=u.AA,
                 z=None, dist=None, meta=None):
        self._wave = np.asarray(wave)
        self._flux = np.asarray(flux)
        self._wunit = wave_unit
        self._unit = unit
        self._z = z
        self._dist = dist

        if error is not None:
            self._error = np.asarray(error)
            if self._wave.shape != self._error.shape:
                raise ValueError('shape of wavelength and variance must match')
        else:
            self._error = None

        if meta is None:
            self.meta = OrderedDict()
        else:
            self.meta = deepcopy(meta)

        if self._wave.shape != self._flux.shape:
            raise ValueError('shape of wavelength and flux must match')
        if self._wave.ndim != 1:
            raise ValueError('only 1-d arrays supported')

    @property
    def wave(self):
        """Wavelength values."""
        return self._wave

    @property
    def flux(self):
        """Spectral flux density values"""
        return self._flux

    @property
    def error(self):
        """Uncertainty on flux density."""
        return self._error

    @property
    def wave_unit(self):
        """Units of wavelength."""
        return self._wunit

    @property
    def unit(self):
        """Units of flux density."""
        return self._unit

    @property
    def z(self):
        """Redshift of spectrum."""
        return self._z

    @z.setter
    def z(self, value):
        self._z = value

    @property
    def dist(self):
        """Distance to object in Mpc."""
        return self._dist

    @dist.setter
    def dist(self, value):
        self._dist = value

    def bandflux(self, band):
        """Perform synthentic photometry in a given bandpass.

        The bandpass transmission is interpolated onto the wavelength grid
        of the spectrum. The result is a weighted sum of the spectral flux
        density values (weighted by transmission values).

        Parameters
        ----------
        band : Bandpass object or name of registered bandpass.

        Returns
        -------
        bandflux : float
            Total flux in ph/s/cm^2. If part of bandpass falls
            outside the spectrum, `None` is returned instead.
        bandfluxerr : float
            Error on flux. Only returned if the `error` attribute is not
            `None`.
        """

        band = get_bandpass(band)
        bwave, btrans = band.to_unit(self._wunit)

        if (bwave[0] < self._wave[0] or bwave[-1] > self._wave[-1]):
            return None

        mask = ((self._wave > bwave[0]) & (self._wave < bwave[-1]))
        d = self._wave[mask]
        f = self._flux[mask]

        # First convert to ergs/s/cm^2/(wavelength unit)...
        target_unit = u.erg / u.s / u.cm**2 / self._wunit
        if self._unit != target_unit:
            f = self._unit.to(target_unit, f,
                              u.spectral_density(self._wunit, d))

        # Then convert ergs to photons: photons = Energy / (h * nu).
        f = f / const.h.cgs.value / self._wunit.to(u.Hz, d, u.spectral())

        trans = np.interp(d, bwave, btrans)
        binw = np.gradient(d)
        ftot = np.sum(f * trans * binw)

        if self._error is None:
            return ftot

        else:
            e = self._error[mask]

            # Do the same conversion as above
            if self._unit != target_unit:
                e = self._unit.to(target_unit, e,
                                  u.spectral_density(self._wunit, d))
            e = e / const.h.cgs.value / self._wunit.to(u.Hz, d, u.spectral())
            etot = np.sqrt(np.sum((e * binw) ** 2 * trans))
            return ftot, etot

    def redshifted_to(self, z, adjust_flux=False, dist=None, cosmo=None):
        """Return a new Spectrum object at a new redshift.

        The current redshift must be defined (self.z cannot be `None`).
        A factor of (1 + z) / (1 + self.z) is applied to the wavelength.
        The inverse factor is applied to the flux so that the bolometric
        flux (e.g., erg/s/cm^2) remains constant.

        .. note:: Currently this only works for units in ergs.

        Parameters
        ----------
        z : float
            Target redshift.
        adjust_flux : bool, optional
            If True, the bolometric flux is adjusted by
            ``F_out = F_in * (D_in / D_out) ** 2``, where ``D_in`` and
            ``D_out`` are current and target luminosity distances,
            respectively. ``D_in`` is self.dist. If self.dist is ``None``,
            the distance is calculated from the current redshift and
            given cosmology.
        dist : float, optional
            Output distance in Mpc. Used to adjust bolometric flux if
            ``adjust_flux`` is ``True``. Default is ``None`` which means
            that the distance is calculated from the redshift and the
            cosmology.
        cosmo : `~astropy.cosmology.Cosmology` instance, optional
            The cosmology used to estimate distances if dist is not given.
            Default is ``None``, which results in using the default
            cosmology.

        Returns
        -------
        spec : Spectrum object
            A new spectrum object at redshift z.
        """

        if self._z is None:
            raise ValueError('Must set current redshift in order to redshift'
                             ' spectrum')

        if self._wunit.physical_type == u.m.physical_type:
            factor = (1. + z) / (1. + self._z)
        elif self._wunit.physical_type == u.Hz.physical_type:
            factor = (1. + self._z) / (1. + z)
        else:
            raise ValueError('wavelength must be in wavelength or frequency')

        d = self._wave * factor
        f = self._flux / factor
        if self._error is not None:
            e = self._error / factor
        else:
            e = None

        if adjust_flux:
            if self._dist is None and self._z == 0.:
                raise ValueError("When current redshift is 0 and adjust_flux "
                                 "is requested, current distance must be "
                                 "defined")
            if dist is None and z == 0.:
                raise ValueError("When redshift is 0 and adjust_flux "
                                 "is requested, dist must be defined")
            if cosmo is None:
                cosmo = cosmology.get_current()

            if self._dist is None:
                dist_in = cosmo.luminosity_distance(self._z)
            else:
                dist_in = self._dist

            if dist is None:
                dist = cosmo.luminosity_distance(z)

            if dist_in <= 0. or dist <= 0.:
                raise ValueError("Distances must be greater than 0.")

            # Adjust the flux
            factor = (dist_in / dist) ** 2
            f *= factor
            if e is not None:
                e *= factor

        return Spectrum(d, f, error=e, z=z, dist=dist, meta=self.meta,
                        unit=self._unit, wave_unit=self._wunit)
