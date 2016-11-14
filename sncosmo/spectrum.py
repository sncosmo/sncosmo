# Licensed under a 3-clause BSD style license - see LICENSE.rst

from copy import deepcopy
from collections import OrderedDict

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy import cosmology
from scipy.interpolate import splrep, splev

from .bandpasses import get_bandpass, HC_ERG_AA
from .models import _integration_grid
from .utils import warn_once

__all__ = ['Spectrum']

SPECTRUM_BANDFLUX_SPACING = 1.0
FLAMBDA_UNIT = u.erg / u.s / u.cm**2 / u.AA


class Spectrum(object):
    """A spectrum, representing wavelength and spectral density values.

    Parameters
    ----------
    wave : list_like
        Wavelength values.
    flux : list_like
        Spectral flux density values.
    wave_unit : `~astropy.units.Unit`
        Unit on wavelength.
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
        self.wave = np.asarray(wave, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        if self.wave.shape != self.flux.shape:
            raise ValueError('shape of wavelength and flux must match')
        if self.wave.ndim != 1:
            raise ValueError('only 1-d arrays supported')

        # internally, wavelength is in Angstroms:
        if wave_unit != u.AA:
            self.wave = wave_unit.to(u.AA, self.wave, u.spectral())
        self._wave_unit = u.AA

        # internally, flux is in F_lambda:
        if unit != FLAMBDA_UNIT:
            self.flux = unit.to(FLAMBDA_UNIT, self.flux,
                                u.spectral_density(u.AA, self.wave))
        self._unit = FLAMBDA_UNIT

        # Set up interpolation.
        # This appears to be the fastest-evaluating interpolant in
        # scipy.interpolate.
        self._tck = splrep(self.wave, self.flux, k=1)

        # following are deprecated attributes:

        if z is not None:
            warn_once("z keyword in Spectrum", "1.4", "2.0")
        self._z = z

        if dist is not None:
            warn_once("dist keyword in Spectrum", "1.4", "2.0")
        self._dist = dist

        if error is not None:
            warn_once("error keyword in Spectrum", "1.4", "2.0")
            self._error = np.asarray(error)
            if self.wave.shape != self._error.shape:
                raise ValueError('shape of wavelength and variance must match')
        else:
            self._error = None

        if meta is None:
            self.meta = OrderedDict()
        else:
            warn_once("meta keyword in Spectrum", "1.4", "2.0")
            self.meta = deepcopy(meta)

    @property
    def error(self):
        """Uncertainty on flux density."""
        warn_once("Spectrum.error", "1.4", "2.0")
        return self._error

    @property
    def wave_unit(self):
        """Units of wavelength."""
        warn_once("Spectrum.wave_unit", "1.4", "2.0")
        return self._wave_unit

    @property
    def unit(self):
        """Units of flux density."""
        warn_once("Spectrum.unit", "1.4", "2.0")
        return self._unit

    @property
    def z(self):
        """Redshift of spectrum."""
        warn_once("Spectrum.z", "1.4", "2.0")
        return self._z

    @z.setter
    def z(self, value):
        warn_once("Spectrum.z", "1.4", "2.0")
        self._z = value

    @property
    def dist(self):
        """Distance to object in Mpc."""
        warn_once("Spectrum.dist", "1.4", "2.0")
        return self._dist

    @dist.setter
    def dist(self, value):
        warn_once("Spectrum.dist", "1.4", "2.0")
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

        # TODO: There is some duplication between this method and
        # models._bandflux_single.

        band = get_bandpass(band)

        # Check that bandpass wavelength range is fully contained in spectrum
        # wavelength range.
        if (band.minwave() < self.wave[0] or band.maxwave() > self.wave[-1]):
            raise ValueError('bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                             'outside spectral range [{3:.6g}, .., {4:.6g}]'
                             .format(band.name, band.minwave(), band.maxwave(),
                                     self.wave[0], self.wave[-1]))

        # Set up wavelength grid. Spacing (dwave) evenly divides the bandpass,
        # closest to 5 angstroms without going over.
        wave, dwave = _integration_grid(band.minwave(), band.maxwave(),
                                        SPECTRUM_BANDFLUX_SPACING)
        trans = band(wave)
        f = splev(wave, self._tck, ext=1)

        return np.sum(wave * trans * f) * dwave / HC_ERG_AA

    def redshifted_to(self, z, adjustflux=False, dist=None, cosmo=None):
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

        warn_once("Spectrum.redshifted_to", "1.4", "2.0")

        if self._z is None:
            raise ValueError('Must set current redshift in order to redshift'
                             ' spectrum')

        if self._wave_unit.physical_type == u.m.physical_type:
            factor = (1. + z) / (1. + self._z)
        elif self._wave_unit.physical_type == u.Hz.physical_type:
            factor = (1. + self._z) / (1. + z)
        else:
            raise ValueError('wavelength must be in wavelength or frequency')

        d = self.wave * factor
        f = self.flux / factor
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
                        unit=self._unit, wave_unit=self._wave_unit)
