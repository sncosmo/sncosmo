# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import astropy.units as u
from scipy.interpolate import splrep, splev

from .bandpasses import get_bandpass
from .constants import SPECTRUM_BANDFLUX_SPACING, HC_ERG_AA
from .magsystems import get_magsystem
from .utils import integration_grid

__all__ = ['Spectrum']

FLAMBDA_UNIT = u.erg / u.s / u.cm**2 / u.AA


def _estimate_bin_edges(wave):
    """Estimate the edges of a set of wavelength bins given the bin centers.

    This function is designed to work for standard linear binning along with
    other more exotic forms of binning such as logarithmic bins. We do a second
    order correction to try to get the bin widths as accurately as possible.

    For linear binning there is only machine precision error with either a
    first or second order estimate.

    For higher order binnings (eg: log), the fractional error is of order (dA /
    A)**2 for linear estimate and (dA / A)**4 for the second order estimate
    that we do here.

    Parameters
    ----------
    wave : `~numpy.ndarray`
        Wavelength values.

    Returns
    -------
    bin_starts : `~numpy.ndarray`
        The estimated start of each wavelength bin.
    bin_ends : `~numpy.ndarray`
        The estimated end of each wavelength bin.
    """

    # First order estimate
    o1 = (wave[:-1] + wave[1:]) / 2.

    # Second order correction
    o2 = 1.5*o1[1:-1] - (o1[2:] + o1[:-2]) / 4.

    # Estimate front and back edges
    f2 = 2 * wave[1] - o2[0]
    f1 = 2 * wave[0] - f2
    b2 = 2 * wave[-2] - o2[-1]
    b1 = 2 * wave[-1] - b2

    # Stack everything together
    bin_edges = np.hstack([f1, f2, o2, b2, b1])
    bin_starts = bin_edges[:-1]
    bin_ends = bin_edges[1:]

    return bin_starts, bin_ends


class Spectrum(object):
    """A spectrum, representing wavelength and spectral density values.

    Parameters
    ----------
    wave : list_like
        Wavelength values.
    flux : list_like
        Spectral flux density values.
    fluxerr : list_like (optional)
        Uncertainties on the flux density values.
    wave_unit : `~astropy.units.Unit`
        Unit on wavelength.
    unit : `~astropy.units.BaseUnit`
        For now, only units with flux density in energy (not photon counts).
    """

    def __init__(self, wave, flux, fluxerr=None, wave_unit=u.AA,
                 unit=(u.erg / u.s / u.cm**2 / u.AA)):
        self.wave = np.asarray(wave, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        if fluxerr is None:
            self.fluxerr = None
        else:
            self.fluxerr = np.asarray(fluxerr, dtype=np.float64)

        if self.wave.shape != self.flux.shape:
            raise ValueError('shape of wavelength and flux must match')
        if self.fluxerr is not None and self.fluxerr.shape != self.flux.shape:
            raise ValueError('shape of flux and fluxerr must match')
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
            if self.fluxerr is not None:
                self.fluxerr = unit.to(FLAMBDA_UNIT, self.fluxerr,
                                       u.spectral_density(u.AA, self.wave))

        self._unit = FLAMBDA_UNIT

        # Set up interpolation.
        # This appears to be the fastest-evaluating interpolant in
        # scipy.interpolate.
        self._tck = splrep(self.wave, self.flux, k=1)
        if self.fluxerr is None:
            self._tck_var = None
        else:
            # For the variance, we need to take the bin widths into account if
            # we want to do interpolation. We estimate the bin widths from the
            # wavelength list.
            bin_starts, bin_ends = _estimate_bin_edges(self.wave)
            bin_widths = bin_ends - bin_starts
            self._tck_var = splrep(self.wave, self.fluxerr**2 * bin_widths,
                                   k=1)

    def bandflux(self, band, zp, zpsys, uncertainty=False):
        """Perform synthentic photometry in a given bandpass.

        The bandpass transmission is interpolated onto the wavelength grid
        of the spectrum. The result is a weighted sum of the spectral flux
        density values (weighted by transmission values).

        Parameters
        ----------
        band : Bandpass or str
            Bandpass object or name of registered bandpass.
        uncertainty : bool
            If False, only the flux is returned. If True, this returns both the
            total band flux and the uncertainty on this total band flux. Note
            that the band flux uncertainty can only be evaluated if the
            uncertainty on the spectrum (fluxerr) is available.

        Returns
        -------
        flux : float
            Total flux in ph/s/cm^2.
        fluxerr : float (optional)
            Statistical uncertainty on the total flux in ph/s/cm^2. Only
            returned if called with uncertainty=True.
        """
        band = get_bandpass(band)

        # Check that bandpass wavelength range is fully contained in spectrum
        # wavelength range.
        if (band.minwave() < self.wave[0] or band.maxwave() > self.wave[-1]):
            raise ValueError('bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                             'outside spectral range [{3:.6g}, .., {4:.6g}]'
                             .format(band.name, band.minwave(), band.maxwave(),
                                     self.wave[0], self.wave[-1]))

        # Check that we have flux uncertainty information if we are trying to
        # evaluate the uncertainty.
        if uncertainty and self.fluxerr is None:
            raise ValueError("need to have the uncertainty on the spectrum to "
                             "evaluate the uncertainty on the band flux")

        # Set up wavelength grid. Spacing (dwave) evenly divides the bandpass,
        # closest to 5 angstroms without going over.
        wave, dwave = integration_grid(band.minwave(), band.maxwave(),
                                       SPECTRUM_BANDFLUX_SPACING)
        trans = band(wave)

        # Evaluate the zeropoint flux
        if zp is None:
            zpnorm = 1.
        else:
            ms = get_magsystem(zpsys)
            zpnorm = 10.**(0.4 * zp) / ms.zpbandflux(band)

        f = splev(wave, self._tck, ext=1)
        total_flux = np.sum(wave * trans * f) * dwave / HC_ERG_AA * zpnorm

        if not uncertainty:
            return total_flux

        # Evaluate the uncertainty on the total flux.
        var = splev(wave, self._tck_var, ext=1)
        total_var = np.sum(wave**2 * trans**2 * var) * dwave / HC_ERG_AA**2
        total_err = np.sqrt(total_var) * zpnorm

        return total_flux, total_err
