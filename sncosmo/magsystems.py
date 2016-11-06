# Licensed under a 3-clause BSD style license - see LICENSE.rst

import abc
import math

import numpy as np
import astropy.units as u
import astropy.constants as const

from ._registry import Registry
from .bandpasses import get_bandpass

__all__ = ['get_magsystem', 'MagSystem', 'SpectralMagSystem',
           'ABMagSystem', 'CompositeMagSystem']

_MAGSYSTEMS = Registry()


def get_magsystem(name):
    """Get a MagSystem from the registry by name."""
    if isinstance(name, MagSystem):
        return name
    return _MAGSYSTEMS.retrieve(name)


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
