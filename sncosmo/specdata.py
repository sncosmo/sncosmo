# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Convenience functions for manipulating spectra."""

import numpy as np
from astropy.table import Table

from .bandpasses import Bandpass
from .photdata import PhotometricData

__all__ = ['SpectrumData']


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
    bin_starts = bin_edges[:-1]
    bin_ends = bin_edges[1:]

    return bin_starts, bin_ends


def _parse_wavelength_data(wave, bin_edges, bin_starts, bin_ends):
    """Parse wavelength data and return a set of bin starts and ends

    TODO: documentation
    """
    # Make sure that a valid combination of inputs was given.
    valid_count = 0
    if wave is not None:
        valid_count += 1
    if bin_edges is not None:
        valid_count += 1
    if bin_starts is not None and bin_ends is not None:
        valid_count += 1
    if valid_count != 1:
        raise ValueError('must specify exactly one of wave, bin_edges, or '
                         'bin_starts and bin_ends')

    # Extract the bin starts and ends.
    if wave is not None:
        bin_starts, bin_ends = _estimate_bin_edges(np.asarray(wave))
    elif bin_edges is not None:
        bin_starts = bin_edges[:-1]
        bin_ends = bin_edges[1:]
    elif bin_starts is None or bin_edges is None:
        raise ValueError('must specify both bin_starts and bin_ends')

    return bin_starts, bin_ends


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
    def __init__(self, wave=None, flux=None, fluxerr=None, bin_edges=None,
                 bin_starts=None, bin_ends=None, time=None):
        # Extract the bin starts and ends
        bin_starts, bin_ends = _parse_wavelength_data(
            wave, bin_edges, bin_starts, bin_ends
        )
        self.bin_starts = bin_starts
        self.bin_ends = bin_ends

        # Make sure that the flux data matches up with the wavelength data.
        self.flux = np.asarray(flux)
        self.fluxerr = np.asarray(fluxerr)
        if not (len(self.bin_starts) == len(self.bin_ends) == len(self.flux) ==
                len(self.fluxerr)):
            raise ValueError("unequal column lengths")

        self.time = time

    @property
    def wave(self):
        """Return the centers of each bin."""
        return (self.bin_starts + self.bin_ends) / 2.

    def get_bands(self):
        """Return a list of bandpass objects for each wavelength element."""

        dwave = 0.00001
        bands = []

        for bin_start, bin_end in zip(self.bin_starts, self.bin_ends):
            bands.append(Bandpass(
                [bin_start, bin_start + dwave, bin_end, bin_end + dwave],
                [0., 1., 1., 0.],
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

    def __len__(self):
        return len(self.flux)
