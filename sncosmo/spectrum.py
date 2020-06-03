# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Convenience functions for interfacing with spectra."""

from astropy.table import Table
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
import astropy.units as u
import numpy as np

from .bandpasses import Bandpass, get_bandpass
from .constants import HC_ERG_AA, SPECTRUM_BANDFLUX_SPACING, FLAMBDA_UNIT
from .magsystems import get_magsystem
from .photdata import PhotometricData
from .utils import integration_grid

__all__ = ['Spectrum']


def _recover_bin_edges(wave):
    """Recover the edges of a set of wavelength bins given the bin centers.

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
        bin_edges = _recover_bin_edges(np.asarray(wave))

    # Make sure that the bin ends are larger than the bin starts.
    if np.any(bin_edges[1:] <= bin_edges[:-1]):
        raise ValueError('wavelength must be monotonically increasing')

    return bin_edges


class Spectrum(object):
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
                 wave_unit=u.AA, unit=FLAMBDA_UNIT, time=None):
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

        # Internally, wavelength is in Angstroms:
        if wave_unit != u.AA:
            self.bin_edges = wave_unit.to(u.AA, self.bin_edges, u.spectral())
        self._wave_unit = u.AA

        # Internally, flux is in F_lambda:
        if unit != FLAMBDA_UNIT:
            unit_scale = unit.to(FLAMBDA_UNIT,
                                 equivalencies=u.spectral_density(u.AA, self.wave))
            self.flux = unit_scale * self.flux
            if self._fluxerr is not None:
                self._fluxerr = unit_scale * self._fluxerr
            if self._fluxcov is not None:
                self._fluxcov = np.outer(unit_scale, unit_scale).dot(self._fluxcov)
        self._unit = FLAMBDA_UNIT

        self.time = time

        # We use a sampling matrix to evaluate models/bands for the spectrum. This
        # matrix is expensive to compute but rarely changes, so we cache it.
        self._cache_sampling_matrix = None

    def __len__(self):
        return len(self.flux)

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

    def has_uncertainties(self):
        """Check whether there is uncertainty information available."""
        return self._fluxcov is not None or self._fluxerr is not None

    def _get_sampling_matrix(self):
        """Build an appropriate sampling for the spectral elements.

        TODO documentation: This returns the wavelengths to sample at along with a
        matrix that converts everything and the dwave.

        TODO: cache the results?
        """
        # Check if we have cached the sampling matrix already.
        if self._cache_sampling_matrix is not None:
            cache_bin_edges, sampling_matrix_result = self._cache_sampling_matrix
            if np.all(cache_bin_edges == self.bin_edges):
                # No changes to the spectral elements so the sampling matrix hasn't
                # changed.
                return sampling_matrix_result

        indices = []
        sample_wave = []
        sample_dwave = []

        for bin_idx in range(len(self.flux)):
            bin_start = self.bin_starts[bin_idx]
            bin_end = self.bin_ends[bin_idx]

            bin_wave, bin_dwave = integration_grid(bin_start, bin_end,
                                                   SPECTRUM_BANDFLUX_SPACING)

            indices.append(bin_idx * np.ones_like(bin_wave, dtype=int))
            sample_wave.append(bin_wave)
            sample_dwave.append(bin_dwave * np.ones_like(bin_wave))

        indices = np.hstack(indices)
        sample_wave = np.hstack(sample_wave)
        sample_dwave = np.hstack(sample_dwave)

        sampling_matrix = csr_matrix(
            (np.ones_like(indices), (indices, np.arange(len(indices)))),
            shape=(len(self), len(indices)),
            dtype=np.float64,
        )

        # Cache the result
        sampling_matrix_result = (sampling_matrix, sample_wave, sample_dwave)
        self._cache_sampling_matrix = (self.bin_edges.copy(), sampling_matrix_result)

        return sampling_matrix_result

    def _band_weights(self, band, zp, zpsys):
        """Calculate the weights for each spectral element for synthetic photometry.

        Parameters
        ----------
        band : `~sncosmo.Bandpass`, str or list_like
            Bandpass, name of bandpass in registry, or list or array thereof.

        Returns
        -------
        band_weights : numpy.array
            The weights to multiply each bin by for synthetic photometry in the given
            band(s). This has a shape of (number of bands, number of spectral elements).
            The dot product of this array with the flux array gives the desired band
            flux.
        """
        band_weights = []

        if zp is not None and zpsys is None:
            raise ValueError('zpsys must be given if zp is not None')

        # broadcast arrays
        if zp is None:
            band = np.atleast_1d(band)
        else:
            band, zp, zpsys = np.broadcast_arrays(np.atleast_1d(band), zp, zpsys)

        for idx in range(len(band)):
            iter_band = get_bandpass(band[idx])

            # Check that bandpass wavelength range is fully contained in spectrum
            # wavelength range.
            if (iter_band.minwave() < self.bin_starts[0]
                    or iter_band.maxwave() > self.bin_ends[-1]):
                raise ValueError('bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                                 'outside spectral range [{3:.6g}, .., {4:.6g}]'
                                 .format(iter_band.name, iter_band.minwave(),
                                         iter_band.maxwave(), self.bin_starts[0],
                                         self.bin_ends[-1]))

            sampling_matrix, sample_wave, sample_dwave = self._get_sampling_matrix()
            trans = iter_band(sample_wave)
            sample_weights = sample_wave * trans * sample_dwave / HC_ERG_AA
            row_band_weights = sampling_matrix * sample_weights

            if zp is not None:
                ms = get_magsystem(zpsys[idx])
                zp_bandflux = ms.zpbandflux(iter_band)
                zpnorm = 10. ** (0.4 * zp[idx]) / zp_bandflux
                row_band_weights *= zpnorm

            band_weights.append(row_band_weights)

        band_weights = np.vstack(band_weights)

        return band_weights

    def bandflux(self, band, zp=None, zpsys=None):
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
        ndim = np.ndim(band)

        band_weights = self._band_weights(band, zp, zpsys)
        band_flux = band_weights.dot(self.flux)

        if ndim == 0:
            band_flux = band_flux[0]

        return band_flux

    def bandfluxcov(self, band, zp=None, zpsys=None):
        """Like bandflux(), but also returns model covariance on values.

        Parameters
        ----------
        band : `~sncosmo.bandpass` or str or list_like
            Bandpass(es) or name(s) of bandpass(es) in registry.
        time : float or list_like
            time(s) in days.
        zp : float or list_like, optional
            If given, zeropoint to scale flux to. if `none` (default) flux
            is not scaled.
        zpsys : `~sncosmo.magsystem` or str (or list_like), optional
            Determines the magnitude system of the requested zeropoint.
            cannot be `none` if `zp` is not `none`.

        Returns
        -------
        bandflux : float or `~numpy.ndarray`
            Model bandfluxes.
        cov : float or `~numpy.array`
            Covariance on ``bandflux``. If ``bandflux`` is an array, this
            will be a 2-d array.
        """
        ndim = np.ndim(band)

        band_weights = self._band_weights(band, zp, zpsys)
        band_flux = band_weights.dot(self.flux)
        band_cov = band_weights.dot(self.fluxcov).dot(band_weights.T)

        if ndim == 0:
            band_flux = band_flux[0]
            band_cov = band_cov[0, 0]

        return band_flux, band_cov

    def bandmag(self, band, magsys):
        """Magnitude at the given phase(s) through the given
        bandpass(es), and for the given magnitude system(s).

        TODO: docs

        Parameters
        ----------
        band : str or list_like
            Name(s) of bandpass in registry.
        magsys : str or list_like
            Name(s) of `~sncosmo.MagSystem` in registry.
        phase : float or list_like
            Phase(s) in days.

        Returns
        -------
        mag : float or `~numpy.ndarray`
            Magnitude for each item in band, magsys, phase.
            The return value is a float if all parameters are not iterables.
            The return value is an `~numpy.ndarray` if any are iterable.
        """
        return -2.5 * np.log10(self.bandflux(band, 0., magsys))
