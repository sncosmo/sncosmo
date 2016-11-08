# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from scipy.interpolate import splrep, splev

from astropy.utils import lazyproperty
from astropy.io import ascii
import astropy.units as u
import astropy.constants as const

from ._registry import Registry
from ._deprecated import warn_once

__all__ = ['get_bandpass', 'read_bandpass', 'Bandpass']

HC_ERG_AA = const.h.cgs.value * const.c.to(u.AA / u.s).value

_BANDPASSES = Registry()


def get_bandpass(name):
    """Get a Bandpass from the registry by name."""
    if isinstance(name, Bandpass):
        return name
    return _BANDPASSES.retrieve(name)


def read_bandpass(fname, fmt='ascii', wave_unit=u.AA,
                  trans_unit=u.dimensionless_unscaled,
                  normalize=False, name=None):
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
    normalize : bool, optional
        If True, normalize fractional transmission to be 1.0 at peak.
        It is recommended to set to True if transmission is in units
        of inverse energy. (When transmission is given in these units, the
        absolute value is usually not significant; normalizing gives more
        reasonable transmission values.) Default is False.
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
                    trans_unit=trans_unit, normalize=normalize,
                    name=name)


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
    normalize : bool, optional
        If True, normalize fractional transmission to be 1.0 at peak.
        It is recommended to set normalize=True if transmission is in units
        of inverse energy. (When transmission is given in these units, the
        absolute value is usually not significant; normalizing gives more
        reasonable transmission values.) Default is False.
    name : str, optional
        Identifier. Default is `None`.

    Examples
    --------
    Construct a Bandpass and access the input arrays:

    >>> b = Bandpass([4000., 4200., 4400.], [0.5, 1.0, 0.5])
    >>> b.wave
    array([ 4000.,  4200.,  4400.])
    >>> b.trans
    array([ 0.5,  1. ,  0.5])

    Bandpasses act like continuous 1-d functions (linear interpolation is
    used):

    >>> b([4100., 4300.])
    array([ 0.75,  0.75])

    The effective (transmission-weighted) wavelength is a property:

    >>> b.wave_eff
    4200.0
    """

    def __init__(self, wave, trans, wave_unit=u.AA,
                 trans_unit=u.dimensionless_unscaled, normalize=False,
                 name=None):
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

        if normalize:
            trans = trans / np.max(trans)

        self.wave = wave
        self.trans = trans

        # Set up interpolation.
        # This appears to be the fastest-evaluating interpolant in
        # scipy.interpolate.
        self._tck = splrep(self.wave, self.trans, k=1)

        self.name = name

    def minwave(self):
        return self.wave[0]

    def maxwave(self):
        return self.wave[-1]

    @lazyproperty
    def dwave(self):
        warn_once("Bandpass.dwave", "1.4", "2.0",
                  "Use numpy.gradient(wave) with your own wavelength array.")
        return np.gradient(self.wave)

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

    def __call__(self, wave):
        return splev(wave, self._tck, ext=1)

    def __repr__(self):
        name = ''
        if self.name is not None:
            name = ' {0!r:s}'.format(self.name)
        return "<Bandpass{0:s} at 0x{1:x}>".format(name, id(self))
