# Licensed under a 3-clause BSD style license - see LICENSE.rst

import abc
import math
from copy import deepcopy
from collections import OrderedDict

import numpy as np

from astropy.utils import lazyproperty
from astropy.io import ascii
import astropy.units as u
import astropy.constants as const
from astropy import cosmology

from ._registry import Registry

__all__ = ['get_bandpass', 'get_magsystem', 'read_bandpass', 'Bandpass',
           'Spectrum', 'MagSystem', 'SpectralMagSystem', 'ABMagSystem',
           'CompositeMagSystem']

HC_ERG_AA = const.h.cgs.value * const.c.to(u.AA / u.s).value

_BANDPASSES = Registry()
_MAGSYSTEMS = Registry()


def get_bandpass(name):
    """Get a Bandpass from the registry by name."""
    if isinstance(name, Bandpass):
        return name
    return _BANDPASSES.retrieve(name)


def get_magsystem(name):
    """Get a MagSystem from the registry by name."""
    if isinstance(name, MagSystem):
        return name
    return _MAGSYSTEMS.retrieve(name)


def read_bandpass(fname, fmt='ascii', wave_unit=u.AA,
                  trans_unit=u.dimensionless_unscaled, name=None):
    """Read bandpass from two-column ASCII file containing wavelength and
    transmission in each line.

    Parameters
    ----------
    fname : str
        File name.
    fmt : {'ascii'}
        File format of file. Currently only ASCII file supported.
    wave_unit : `~astropy.units.Unit` or str, optional
        Wavelength unit. Default is Angstroms.
    trans_unit : `~astropy.units.Unit`, optional
        Transmission unit. Can be `~astropy.units.dimensionless_unscaled`,
        indicating a ratio of transmitted to incident photons, or units
        proportional to inverse energy, indicating a ratio of transmitted
        photons to incident energy. Default is ratio of transmitted to
        incident photons.
    name : str, optional
        Identifier. Default is `None`.

    Returns
    -------
    band : `~sncosmo.Bandpass`
    """

    if fmt != 'ascii':
        raise ValueError("format {0} not supported. Supported formats: 'ascii'"
                         .format(fmt))
    t = ascii.read(fname, names=['wave', 'trans'])
    return Bandpass(t['wave'], t['trans'], wave_unit=wave_unit,
                    trans_unit=trans_unit, name=name)


class Bandpass(object):
    """Transmission as a function of spectral wavelength.

    Parameters
    ----------
    wave : list_like
        Wavelength. Monotonically increasing values.
    trans : list_like
        Transmission fraction.
    wave_unit : `~astropy.units.Unit` or str, optional
        Wavelength unit. Default is Angstroms.
    trans_unit : `~astropy.units.Unit`, optional
        Transmission unit. Can be `~astropy.units.dimensionless_unscaled`,
        indicating a ratio of transmitted to incident photons, or units
        proportional to inverse energy, indicating a ratio of transmitted
        photons to incident energy. Default is ratio of transmitted to
        incident photons.
    name : str, optional
        Identifier. Default is `None`.

    Examples
    --------
    >>> b = Bandpass([4000., 4200., 4400.], [0.5, 1.0, 0.5])
    >>> b.wave
    array([ 4000.,  4200.,  4400.])
    >>> b.trans
    array([ 0.5,  1. ,  0.5])
    >>> b.dwave
    array([ 200.,  200.,  200.])
    >>> b.wave_eff
    4200.0

    """

    def __init__(self, wave, trans, wave_unit=u.AA,
                 trans_unit=u.dimensionless_unscaled, name=None):
        wave = np.asarray(wave, dtype=np.float64)
        trans = np.asarray(trans, dtype=np.float64)
        if wave.shape != trans.shape:
            raise ValueError('shape of wave and trans must match')
        if wave.ndim != 1:
            raise ValueError('only 1-d arrays supported')

        # Ensure that units are actually units and not quantities, so that
        # `to` method returns a float and not a Quantity.
        wave_unit = u.Unit(wave_unit)
        trans_unit = u.Unit(trans_unit)

        if wave_unit != u.AA:
            wave = wave_unit.to(u.AA, wave, u.spectral())

        # If transmission is in units of inverse energy, convert to
        # unitless transmission:
        #
        # (transmitted photons / incident photons) =
        #      (photon energy) * (transmitted photons / incident energy)
        #
        # where photon energy = h * c / lambda
        if trans_unit != u.dimensionless_unscaled:
            trans = (HC_ERG_AA / wave) * trans_unit.to(u.erg**-1, trans)

        # Check that values are monotonically increasing.
        # We could sort them, but if this happens, it is more likely a user
        # error or faulty bandpass definition. So we leave it to the user to
        # sort them.
        if not np.all(np.ediff1d(wave) > 0.):
            raise ValueError('bandpass wavelength values must be monotonically'
                             ' increasing when supplied in wavelength or '
                             'decreasing when supplied in energy/frequency.')
        self.wave = wave
        self._dwave = np.gradient(wave)
        self.trans = trans
        self.name = name

    @property
    def dwave(self):
        """Gradient of wavelengths, numpy.gradient(wave)."""
        return self._dwave

    @lazyproperty
    def wave_eff(self):
        """Effective wavelength of bandpass in Angstroms."""
        weights = self.trans * np.gradient(self.wave)
        return np.sum(self.wave * weights) / np.sum(weights)

    def to_unit(self, unit):
        """Return wavelength and transmission in new wavelength units.

        If the requested units are the same as the current units, self is
        returned.

        Parameters
        ----------
        unit : `~astropy.units.Unit` or str
            Target wavelength unit.

        Returns
        -------
        wave : `~numpy.ndarray`
        trans : `~numpy.ndarray`
        """

        if unit is u.AA:
            return self.wave, self.trans

        d = u.AA.to(unit, self.wave, u.spectral())
        t = self.trans
        if d[0] > d[-1]:
            d = np.flipud(d)
            t = np.flipud(t)
        return d, t

    def __repr__(self):
        name = ''
        if self.name is not None:
            name = ' {0!r:s}'.format(self.name)
        return "<Bandpass{0:s} at 0x{1:x}>".format(name, id(self))


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


class MagSystem(object):
    """An abstract base class for magnitude systems."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, name=None):
        self._zpbandflux = {}
        self._name = name

    @abc.abstractmethod
    def _refspectrum_bandflux(self, band):
        """Flux of the fundamental spectrophotometric standard."""
        pass

    @property
    def name(self):
        """Name of magnitude system."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def zpbandflux(self, band):
        """Flux of an object with magnitude zero in the given bandpass.

        Parameters
        ----------
        bandpass : `~sncosmo.spectral.Bandpass` or str

        Returns
        -------
        bandflux : float
            Flux in photons / s / cm^2.
        """

        band = get_bandpass(band)
        try:
            return self._zpbandflux[band]
        except KeyError:
            bandflux = self._refspectrum_bandflux(band)
            self._zpbandflux[band] = bandflux

        return bandflux

    def band_flux_to_mag(self, flux, band):
        """Convert flux (photons / s / cm^2) to magnitude."""
        return -2.5 * math.log10(flux / self.zpbandflux(band))

    def band_mag_to_flux(self, mag, band):
        """Convert magnitude to flux in photons / s / cm^2"""
        return self.zpbandflux(band) * 10.**(-0.4 * mag)


class CompositeMagSystem(MagSystem):
    """A magnitude system defined in a specific set of bands.

    In each band, there is a fundamental standard with a known
    (generally non-zero) magnitude.

    Parameters
    ----------
    bands: iterable of `~sncosmo.Bandpass` or str
        The filters in the magnitude system.
    standards: iterable of `~sncosmo.MagSystem` or str,
        The spectrophotmetric flux standards for each band, in the
        same order as `bands`.
    offsets: list_like
        The magnitude of standard in the given band.
    """

    def __init__(self, bands, standards, offsets, name=None):
        super(CompositeMagSystem, self).__init__(name=name)

        if not len(bands) == len(offsets) == len(standards):
            raise ValueError('Lengths of bands, standards, and offsets '
                             'must match.')

        self._bands = [get_bandpass(band) for band in bands]
        self._standards = [get_magsystem(s) for s in standards]
        self._offsets = offsets

    @property
    def bands(self):
        return self._bands

    @property
    def standards(self):
        return self._standards

    @property
    def offsets(self):
        return self._offsets

    def _refspectrum_bandflux(self, band):
        if band not in self._bands:
            raise ValueError('band not in local magnitude system')
        i = self._bands.index(band)
        standard = self._standards[i]
        offset = self._offsets[i]

        return 10.**(0.4 * offset) * standard.zpbandflux(band)

    def __str__(self):
        s = "CompositeMagSystem {!r}:\n".format(self.name)

        for i in range(len(self._bands)):
            s += "  {!r}: system={!r}  offset={}\n".format(
                self._bands[i].name,
                self._standards[i].name,
                self._offsets[i])
        return s


class SpectralMagSystem(MagSystem):
    """A magnitude system defined by a fundamental spectrophotometric
    standard.

    Parameters
    ----------
    refspectrum : `sncosmo.Spectrum`
        The spectrum of the fundamental spectrophotometric standard.
    """

    def __init__(self, refspectrum, name=None):
        super(SpectralMagSystem, self).__init__(name)
        self._refspectrum = refspectrum

    def _refspectrum_bandflux(self, band):
        return self._refspectrum.bandflux(band)


class ABMagSystem(MagSystem):
    """Magnitude system where a source with F_nu = 3631 Jansky at all
    frequencies has magnitude 0 in all bands."""

    def _refspectrum_bandflux(self, band):
        bwave, btrans = band.to_unit(u.Hz)

        # AB spectrum is 3631 x 10^{-23} erg/s/cm^2/Hz
        # Get spectral values in photons/cm^2/s/Hz at bandpass wavelengths
        # by dividing by (h \nu).
        f = 3631.e-23 / const.h.cgs.value / bwave

        binw = np.gradient(bwave)
        return np.sum(f * btrans * binw)
