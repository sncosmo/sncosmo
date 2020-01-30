# Licensed under a 3-clause BSD style license - see LICENSE.rst

import abc
import math

import numpy as np
import astropy.units as u
import astropy.constants as const

from ._registry import Registry
from .bandpasses import get_bandpass
from .utils import integration_grid
from .constants import H_ERG_S, SPECTRUM_BANDFLUX_SPACING

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
    """
    CompositeMagSystem(bands=None, families=None, name=None)

    A magnitude system defined in a specific set of bands.

    In each band, there is a fundamental standard with a known
    (generally non-zero) magnitude.

    Parameters
    ----------
    bands: dict, optional
        Dictionary where keys are `~sncosmo.Bandpass` instances or names,
        thereof and values are 2-tuples of magnitude system and offset.
        The offset gives the magnitude of standard in the given band.
        A positive offset means that the composite magsystem zeropoint flux
        is higher (brighter) than that of the standard.
    families : dict, optional
        Similar to the ``bands`` argument, but keys are strings that apply to
        any bandpass that has a matching ``family`` attribute. This is useful
        for generated bandpasses where the transmission differs across
        focal plane (and hence the bandpass at each position is unique), but
        all photometry has been calibrated to the same offset.
    name : str
        The ``name`` attribute of the magnitude system.

    Examples
    --------
    Create a magnitude system defined in only two SDSS bands where an
    object with AB magnitude of 0 would have a magnitude of 0.01 and 0.02
    in the two bands respectively:

    >>> sncosmo.CompositeMagSystem(bands={'sdssg': ('ab', 0.01),
    ...                                   'sdssr': ('ab', 0.02)})
    """

    def __init__(self, bands=None, families=None, name=None):
        super(CompositeMagSystem, self).__init__(name=name)

        if bands is not None:
            self._bands = {get_bandpass(band): (get_magsystem(magsys), offset)
                           for band, (magsys, offset) in bands.items()}
        else:
            self._bands = {}

        if families is not None:
            self._families = {f: (get_magsystem(magsys), offset)
                              for f, (magsys, offset) in families.items()}
        else:
            self._families = {}

    @property
    def bands(self):
        return self._bands

    def _refspectrum_bandflux(self, band):
        val = self._bands.get(band)

        if val is not None:
            standard, offset = val
            return 10.**(0.4 * offset) * standard.zpbandflux(band)

        if hasattr(band, 'family'):
            val = self._families.get(band.family)
            if val is not None:
                standard, offset = val
                return 10.**(0.4 * offset) * standard.zpbandflux(band)

        raise ValueError('band not defined in composite magnitude system')

    def __str__(self):
        s = "CompositeMagSystem {!r}:\n".format(self.name)

        for band, (magsys, offset) in self._bands.items():
            s += "  {!r}: system={!r}  offset={}\n".format(
                band, magsys, offset)

        for family, (magsys, offset) in self._families.items():
            s += "  {!r}: system={!r}  offset={}\n".format(
                family, magsys, offset)

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
        return self._refspectrum.bandflux(band, None, None)


class ABMagSystem(MagSystem):
    """Magnitude system where a source with F_nu = 3631 Jansky at all
    frequencies has magnitude 0 in all bands."""

    def _refspectrum_bandflux(self, band):
        wave, dwave = integration_grid(band.minwave(), band.maxwave(),
                                       SPECTRUM_BANDFLUX_SPACING)

        # AB spectrum is 3631 x 10^{-23} erg/s/cm^2/Hz
        #
        # In F_lambda this is 3631e-23 * c[AA/s] / wave[AA]**2  erg/s/cm^2/AA
        #
        # To integrate, we do
        #
        # sum(f * trans * wave) * dwave / (hc[erg AA])
        #
        # so we can simplify the above into
        #
        #   sum(3631e-23 * c / wave * trans) * dwave / hc
        # = 3631e-23 * dwave / h[ERG S] * sum(trans / wave)
        return 3631e-23 * dwave / H_ERG_S * np.sum(band(wave) / wave)
