# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Convenience functions for manipulating spectra."""

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
    bin_starts = bin_edges[:-1]
    bin_ends = bin_edges[1:]

    return bin_starts, bin_ends


def _parse_wavelength_information(wave, bin_edges, bin_starts, bin_ends):
    """Parse wavelength information and return a set of bin starts and ends

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
    elif bin_starts is None or bin_ends is None:
        raise ValueError('must specify both bin_starts and bin_ends')

    # Make sure that the bin ends are larger than the bin starts.
    if np.any(bin_starts >= bin_ends):
        raise ValueError('bin_ends must be larger than bin_starts')

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
    def __init__(self, wave=None, flux=None, fluxerr=None, fluxcov=None,
                 bin_edges=None, bin_starts=None, bin_ends=None, time=None):
        # Extract the bin starts and ends
        bin_starts, bin_ends = _parse_wavelength_information(
            wave, bin_edges, bin_starts, bin_ends
        )
        self.bin_starts = bin_starts
        self.bin_ends = bin_ends

        # Make sure that the flux data matches up with the wavelength data.
        self.flux = np.asarray(flux)
        if not (len(self.bin_starts) == len(self.bin_ends) == len(self.flux)):
            raise ValueError("unequal column lengths")

        # Extract uncertainty information in whatever form it came in.
        self._fluxerr = None
        self._fluxcov = None

        if fluxerr is not None:
            if fluxcov is not None:
                raise ValueError("can only specify one of fluxerr and fluxcov")
            self._fluxerr = np.array(fluxerr)
            if len(self._fluxerr) != len(self.bin_starts):
                raise ValueError("unequal column lengths")
        elif fluxcov is not None:
            self._fluxcov = np.array(fluxcov)
            if not (len(self.bin_starts) == self._fluxcov.shape[0] ==
                    self._fluxcov.shape[1]):
                raise ValueError("unequal column lengths")

        self.time = time

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

    def merge(self, other):
        """Merge two spectra.

        This doesn't do any rebinning, it keeps all of the bins from the two
        original spectra. Call .rebin() if you want to get a contiguous
        spectrum.
        """

        # Figure out what to do with the uncertainties.
        if self.has_uncertainties() != other.has_uncertainties():
            raise ValueError("one spectrum has uncertainties but the other "
                             "doesn't. can't handle.")
        elif not (self.has_uncertainties() or other.has_uncertainties()):
            # No uncertainties for either spectrum.
            merge_fluxerr = None
            merge_fluxcov = None
        if self._fluxerr is not None and other._fluxerr is not None:
            # Have uncorrelated uncertainties for both.
            merge_fluxerr = np.concatenate([self._fluxerr, other._fluxerr])
            merge_fluxcov = None
        else:
            # Have a covariance matrix for one. Build a combined covariance
            # matrix assuming that the two spectra have no covariance between
            # them.
            merge_fluxerr = None
            merge_fluxcov = block_diag(self.fluxcov, other.fluxcov)

        return SpectrumData(
            bin_starts=np.concatenate([self.bin_starts, other.bin_starts]),
            bin_ends=np.concatenate([self.bin_ends, other.bin_ends]),
            flux=np.concatenate([self.flux, other.flux]),
            fluxerr=merge_fluxerr,
            fluxcov=merge_fluxcov,
            time=self.time,
        )


    def rebin(self, wave=None, bin_edges=None, bin_starts=None, bin_ends=None):
        """Rebin the spectrum with the given bin edges.

        TODO: variance weighting or things like that?
        """
        new_bin_starts, new_bin_ends = (
            _parse_wavelength_information(wave, bin_edges, bin_starts, bin_ends)
        )

        old_bin_starts = self.bin_starts
        old_bin_ends = self.bin_ends

        # Generate a weight matrix for the transformation.
        overlap_starts = np.max(np.meshgrid(old_bin_starts, new_bin_starts),
                                axis=0)
        overlap_ends = np.min(np.meshgrid(old_bin_ends, new_bin_ends), axis=0)
        overlaps = overlap_ends - overlap_starts
        overlaps[overlaps < 0] = 0

        # Normalize by the total overlap in each bin to keep everything in units
        # of f_lambda
        total_overlaps = np.sum(overlaps, axis=1)
        if np.any(total_overlaps == 0):
            raise ValueError("new binning not contained within original "
                             "spectrum")
        weight_matrix = overlaps / total_overlaps[:, None]

        new_flux = weight_matrix.dot(self.flux)
        new_fluxcov = weight_matrix.dot(self.fluxcov.dot(weight_matrix.T))

        return SpectrumData(
            bin_starts=new_bin_starts,
            bin_ends=new_bin_ends,
            flux=new_flux,
            fluxcov=new_fluxcov,
            time=self.time,
        )

    def is_contiguous(self):
        """Check whether the spectral elements are contiguous."""
        return np.all(self.bin_starts[1:] == self.bin_ends[:-1])

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
        if not self.is_contiguous():
            raise ValueError('spectral elements are not contiguous. rebin '
                             'before calculating synthetic photometry')

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
