# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Convenience functions for interfacing with spectra."""

import numpy as np
from astropy.table import Table
from scipy.linalg import block_diag

from .bandpasses import Bandpass, get_bandpass
from .photdata import PhotometricData
from .constants import HC_ERG_AA, SPECTRUM_BANDFLUX_SPACING
from .utils import integration_grid

__all__ = ['SpectrumData']


def _estimate_bin_edges(wave):
    """Estimate the edges of a set of wavelength bins given the bin centers.

    This function is designed to work for standard linear binning along with
    other more exotic forms of binning such as logarithmic bins. We do a second
    order correction to try to get the bin widths as accurately as possible. For
    linear binning there is only machine precision error with either a first or
    second order estimate.

    For higher order binnings (eg: log), the fractional error is of order (dA /
    A)**2 for linear estimate and (dA / A)**4 for the second order estimate
    that we do here.

    Parameters
    ----------
    wave : array-like
        Central wavelength values of each wavelength bin.
    Returns
    -------
    bin_starts : `~numpy.ndarray`
        The estimated start of each wavelength bin.
    bin_ends : `~numpy.ndarray`
        The estimated end of each wavelength bin.
    """
    wave = np.asarray(wave)

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

    return bin_edges


def _parse_wavelength_information(wave, bin_edges):
    """Parse wavelength information and return a set of bin edges.

    TODO: documentation
    """
    # Make sure that a valid combination of inputs was given.
    valid_count = 0
    if wave is not None:
        valid_count += 1
    if bin_edges is not None:
        valid_count += 1
    if valid_count != 1:
        raise ValueError('must specify exactly one of wave or bin_edges')

    # Extract the bin starts and ends.
    if wave is not None:
        bin_edges = _estimate_bin_edges(np.asarray(wave))

    # Make sure that the bin ends are larger than the bin starts.
    if np.any(bin_edges[1:] <= bin_edges[:-1]):
        raise ValueError('wavelength must be monotonically increasing')

    return bin_edges


class SpectrumData(object):
    """Standardized representation of spectroscopic data.

    The edges of the spectral element wavelength bins are stored in
    ``bin_edges`` as a numpy array. ``flux`` and ``fluxerr`` are the
    corresponding flux values and uncertainties of the spectrum for each
    spectral element, and are both numpy arrays with lengths of one less than
    the size of ``bin_edges``.

    Optionally, there is a single ``time`` associated with all of the
    observations.

    TODO: calibrated spectra? See spectrum.py where there is wave_unit and unit,
    and everything is converted internally.
    TODO: covariance?

    Has attribute ``fluxcov`` which may be ``None``.

    Parameters
    ----------
    TODO

    """
    def __init__(self, wave=None, flux=None, fluxerr=None, fluxcov=None, bin_edges=None,
                 time=None):
        # Extract the bin edges
        bin_edges = _parse_wavelength_information(wave, bin_edges)
        self.bin_edges = bin_edges

        # Make sure that the flux data matches up with the wavelength data.
        self.flux = np.asarray(flux)
        if not (len(self.bin_edges) - 1 == len(self.flux)):
            raise ValueError("unequal column lengths")

        # Extract uncertainty information in whatever form it came in.
        self._fluxerr = None
        self._fluxcov = None

        if fluxerr is not None:
            if fluxcov is not None:
                raise ValueError("can only specify one of fluxerr and fluxcov")
            self._fluxerr = np.array(fluxerr)
            if len(self._fluxerr) != len(self.flux):
                raise ValueError("unequal column lengths")
        elif fluxcov is not None:
            self._fluxcov = np.array(fluxcov)
            if not (len(self.flux) == self._fluxcov.shape[0] == self._fluxcov.shape[1]):
                raise ValueError("unequal column lengths")

        self.time = time

    @property
    def bin_starts(self):
        """Return the start of each bin."""
        return self.bin_edges[:-1]

    @property
    def bin_ends(self):
        """Return the end of each bin."""
        return self.bin_edges[1:]

    @property
    def wave(self):
        """Return the centers of each bin."""
        return (self.bin_starts + self.bin_ends) / 2.

    @property
    def fluxerr(self):
        """Return the uncertainties on each flux bin"""
        if self._fluxerr is not None:
            return self._fluxerr
        elif self._fluxcov is not None:
            return np.sqrt(np.diag(self._fluxcov))
        else:
            raise ValueError("no uncertainty information available")

    @property
    def fluxcov(self):
        """Return the covariance matrix"""
        if self._fluxcov is not None:
            return self._fluxcov
        elif self._fluxerr is not None:
            return np.diag(self._fluxerr**2)
        else:
            raise ValueError("no uncertainty information available")

    def get_bands(self):
        """Return a list of bandpass objects for each wavelength element."""

        bands = []

        for bin_start, bin_end in zip(self.bin_starts, self.bin_ends):
            bands.append(Bandpass(
                [bin_start, bin_end],
                [1., 1.],
            ))

        bands = np.array(bands)

        return bands

    def get_table(self):
        """Convert the spectrum into an `astropy.Table` object"""
        bands = self.get_bands()
        wave = self.wave

        # TODO: move this to the constants file or something. This is the
        # conversion between AB mag and f_lambda
        scale = wave**2 / 3e8 / 1e10 * 10**(0.4 * 48.60)

        photdata = Table({
            'time': np.ones(len(self)) * self.time,
            'band': self.get_bands(),
            'flux': self.flux * scale,
            'fluxerr': self.fluxerr * scale,
            'zp': np.zeros(len(self)),
            'zpsys': np.array(['ab'] * len(self)),
        })

        return photdata

    def has_uncertainties(self):
        """Check whether there is uncertainty information available."""
        return self._fluxcov is not None or self._fluxerr is not None

    def bandflux(self, band):
        """Perform synthentic photometry in a given bandpass.

        The bandpass transmission is interpolated onto the wavelength grid of
        the spectrum. The result is a weighted sum of the spectral flux density
        values (weighted by transmission values).

        TODO docs

        Parameters
        ----------
        band : Bandpass or str
            Bandpass object or name of registered bandpass.

        Returns
        -------
        float
            Total flux in ph/s/cm^2.
        """
        band = get_bandpass(band)

        # Check that bandpass wavelength range is fully contained in spectrum
        # wavelength range.
        if (band.minwave() < self.bin_starts[0] or band.maxwave() >
                self.bin_ends[-1]):
            raise ValueError('bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                             'outside spectral range [{3:.6g}, .., {4:.6g}]'
                             .format(band.name, band.minwave(), band.maxwave(),
                                     self.bin_starts[0], self.bin_ends[-1]))

        # Calculate the weight for each spectral element.
        weights = np.zeros(len(self))
        for bin_idx in range(len(self)):
            wave, dwave = integration_grid(self.bin_starts[bin_idx],
                                           self.bin_ends[bin_idx],
                                           SPECTRUM_BANDFLUX_SPACING)
            trans = band(wave)

            bin_weight = np.sum(wave * trans * dwave) / HC_ERG_AA
            weights[bin_idx] = bin_weight

        # Calculate the flux and uncertainties.
        bandflux = self.flux.dot(weights)
        bandfluxerr = np.sqrt(self.fluxcov.dot(weights).dot(weights))

        return bandflux, bandfluxerr

    def __len__(self):
        return len(self.flux)
