# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from astropy.utils import lazyproperty
from astropy.io import ascii
import astropy.units as u
import astropy.constants as const

from ._registry import Registry

__all__ = ['get_bandpass', 'read_bandpass', 'Bandpass']

HC_ERG_AA = const.h.cgs.value * const.c.to(u.AA / u.s).value

_BANDPASSES = Registry()


def get_bandpass(name):
    """Get a Bandpass from the registry by name."""
    if isinstance(name, Bandpass):
        return name
    return _BANDPASSES.retrieve(name)


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
