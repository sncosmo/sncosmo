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
    bin_edges : `~numpy.ndarray`
        The recovered edges of each wavelength bin.
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

    Either the central wavelength for each bin can be passed as ``wave``, or
    the bin edges can be passed directly as ``bin_edges``. This function will
    recover the bin edges from either input and verify that they are a valid
    monotonically-increasing list.
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
    """An observed spectrum of an object.

    This class is designed to represent an observed spectrum. An observed
    spectrum is a set of contiguous bins in wavelength (referred to as
    "spectral elements") with associated flux measurements. We assume that each
    spectral element has uniform transmission in wavelength. A spectrum can
    optionally have associated uncertainties or covariance between the observed
    fluxes of the different spectral elements. A spectrum can also optionally
    have a time associated with it.

    Internally, we store the edges of each of the spectral element wavelength
    bins. These are automatically recovered in the common case where a user has
    a list of central wavelengths for each bin. The wavelengths are stored
    internally in units of Angstroms. The flux is stored as a spectral flux
    density F_λ (units of erg / s / cm^2 / Angstrom).

    Parameters
    ----------
    wave : list-like
        Central wavelengths of each spectral element. This must be
        monotonically increasing. This is assumed to be in units of Angstroms
        unless ``wave_unit`` is specified.
    flux : list-like
        Observed fluxes for each spectral element. By default this is assumed
        to be a spectral flux density F_λ unless ``unit`` is explicitly
        specified.
    fluxerr : list-like
        Uncertainties on the observed fluxes for each spectral element.
    fluxcov : two-dimensional `~numpy.ndarray`
        Covariance of the observed fluxes for each spectral element. Only one
        of ``fluxerr`` and ``fluxcov`` may be specified.
    bin_edges : list-like
        Edges of each spectral element in wavelength. This should be a list
        that is length one longer than ``flux``. Only one of ``wave`` and
        ``bin_edges`` may be specified.
    wave_unit : `~astropy.units.Unit`
        Wavelength unit. Default is Angstroms.
    unit : `~astropy.units.Unit`
        Flux unit. Default is F_λ (erg / s / cm^2 / Angstrom).
    time : float
        The time associated with the spectrum. This is required if fitting a
        model to the spectrum.
    """
    def __init__(self, wave=None, flux=None, fluxerr=None, fluxcov=None,
                 bin_edges=None, wave_unit=u.AA, unit=FLAMBDA_UNIT, time=None):
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
            if not (len(self.flux) == self._fluxcov.shape[0] ==
                    self._fluxcov.shape[1]):
                raise ValueError("unequal column lengths")

        # Internally, wavelength is in Angstroms:
        if wave_unit != u.AA:
            self.bin_edges = wave_unit.to(u.AA, self.bin_edges, u.spectral())
        self._wave_unit = u.AA

        # Internally, flux is in F_lambda:
        if unit != FLAMBDA_UNIT:
            unit_scale = unit.to(
                FLAMBDA_UNIT, equivalencies=u.spectral_density(u.AA, self.wave)
            )
            self.flux = unit_scale * self.flux
            if self._fluxerr is not None:
                self._fluxerr = unit_scale * self._fluxerr
            if self._fluxcov is not None:
                self._fluxcov = np.outer(unit_scale, unit_scale) \
                    .dot(self._fluxcov)
        self._unit = FLAMBDA_UNIT

        self.time = time

        # We use a sampling matrix to evaluate models/bands for the spectrum.
        # This matrix is expensive to compute but rarely changes, so we cache
        # it.
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

    def rebin(self, wave=None, bin_edges=None):
        """Rebin the spectrum on a new wavelength grid.

        We assume that the spectrum is constant for each spectral element with
        a value given by its observed flux. If the new bin edges are not
        aligned with the old ones, then this will introduce covariance between
        spectral elements. We propagate that covariance properly.

        Parameters
        ----------
        wave : list-like
            Central wavelengths of the rebinned spectrum.
        bin_edges : list-like
            Bin edges of the rebinned spectrum. Only one of ``wave`` and
            ``bin_edges`` may be specified.

        Returns
        -------
        rebinned_spectrum : `~sncosmo.Spectrum`
            A new `~sncosmo.Spectrum` with the rebinned spectrum.
        """
        new_bin_edges = _parse_wavelength_information(wave, bin_edges)
        new_bin_starts = new_bin_edges[:-1]
        new_bin_ends = new_bin_edges[1:]

        old_bin_starts = self.bin_starts
        old_bin_ends = self.bin_ends

        # Generate a weight matrix for the transformation.
        overlap_starts = np.max(np.meshgrid(old_bin_starts, new_bin_starts),
                                axis=0)
        overlap_ends = np.min(np.meshgrid(old_bin_ends, new_bin_ends), axis=0)
        overlaps = overlap_ends - overlap_starts
        overlaps[overlaps < 0] = 0

        # Normalize by the total overlap in each bin to keep everything in
        # units of f_lambda
        total_overlaps = np.sum(overlaps, axis=1)
        if np.any(total_overlaps == 0):
            raise ValueError("new binning not contained within original "
                             "spectrum")
        weight_matrix = overlaps / total_overlaps[:, None]

        new_flux = weight_matrix.dot(self.flux)
        new_fluxcov = weight_matrix.dot(self.fluxcov.dot(weight_matrix.T))

        return Spectrum(
            bin_edges=new_bin_edges,
            flux=new_flux,
            fluxcov=new_fluxcov,
            time=self.time,
        )

    def get_sampling_matrix(self):
        """Build an appropriate sampling for the spectral elements.

        For spectra with wide spectral elements, it is important to integrate
        models over the spectral element rather than simply sampling at the
        central wavelength. This function first determines where to sample for
        each spectral element and returns the corresponding list of wavelengths
        ``sample_wave``. This function also returns a matrix
        ``sampling_matrix`` that provided the mapping between the sampled
        wavelengths and the spectral elements. Given a set of model fluxes
        evaluated at ``sample_wave``, the dot product of ``sampling_matrix``
        with these fluxes gives the corresponding fluxes in each spectral
        element in units of (erg / s / cm^2).

        ``sampling_matrix`` is stored as a compressed sparse row matrix that
        can be very efficiently used for dot products with vectors. This matrix
        is somewhat expensive to calculate and only changes if the bin edges of
        the spectral elements change, so we cache it and only recompute it if
        the bin edges change.

        Returns
        -------
        sample_wave : `~numpy.ndarray`
            Wavelengths to sample a model at.
        sampling_matrix : `~scipy.sparse.csr_matrix`
            Matrix giving the mapping from the sampled bins to the spectral
            elements.
        """
        # Check if we have cached the sampling matrix already.
        if self._cache_sampling_matrix is not None:
            cache_bin_edges, sampling_matrix_result = \
                self._cache_sampling_matrix
            if np.all(cache_bin_edges == self.bin_edges):
                # No changes to the spectral elements so the sampling matrix
                # hasn't changed.
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
            (sample_dwave, (indices, np.arange(len(indices)))),
            shape=(len(self), len(indices)),
            dtype=np.float64,
        )

        # Cache the result
        sampling_matrix_result = (sample_wave, sampling_matrix)
        self._cache_sampling_matrix = (self.bin_edges.copy(),
                                       sampling_matrix_result)

        return sampling_matrix_result

    def _band_weights(self, band, zp, zpsys):
        """Calculate the weights for synthetic photometry.

        Parameters
        ----------
        band : `~sncosmo.Bandpass`, str or list_like
            Bandpass, name of bandpass in registry, or list or array thereof.
        zp : float or list_like, optional
            If given, zeropoint to scale flux to (must also supply ``zpsys``).
            If not given, flux is not scaled.
        zpsys : str or list_like, optional
            Name of a magnitude system in the registry, specifying the system
            that ``zp`` is in.

        Returns
        -------
        band_weights : numpy.array
            The weights to multiply each bin by for synthetic photometry in the
            given band(s). This has a shape of (number of bands, number of
            spectral elements). The dot product of this array with the flux
            array gives the desired band flux.
        """
        band_weights = []

        if zp is not None and zpsys is None:
            raise ValueError('zpsys must be given if zp is not None')

        # broadcast arrays
        if zp is None:
            band = np.atleast_1d(band)
        else:
            band, zp, zpsys = np.broadcast_arrays(np.atleast_1d(band), zp,
                                                  zpsys)

        for idx in range(len(band)):
            iter_band = get_bandpass(band[idx])

            # Check that bandpass wavelength range is fully contained in
            # spectrum wavelength range.
            if (iter_band.minwave() < self.bin_starts[0]
                    or iter_band.maxwave() > self.bin_ends[-1]):
                raise ValueError(
                    'bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                    'outside spectral range [{3:.6g}, .., {4:.6g}]'
                    .format(iter_band.name, iter_band.minwave(),
                            iter_band.maxwave(), self.bin_starts[0],
                            self.bin_ends[-1])
                )

            sample_wave, sampling_matrix = self.get_sampling_matrix()
            trans = iter_band(sample_wave)
            sample_weights = sample_wave * trans / HC_ERG_AA
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

        We assume that the spectrum is constant for each spectral element with
        a value given by its observed flux. The bandpass is sampled on an
        appropriate high-resolution grid and multiplied with the observed
        fluxes to give the corresponding integrated band flux over this band.

        Parameters
        ----------
        band : `~sncosmo.bandpass` or str or list_like
            Bandpass(es) or name(s) of bandpass(es) in registry.
        zp : float or list_like, optional
            If given, zeropoint to scale flux to. if `none` (default) flux
            is not scaled.
        zpsys : `~sncosmo.magsystem` or str (or list_like), optional
            Determines the magnitude system of the requested zeropoint.
            cannot be `none` if `zp` is not `none`.

        Returns
        -------
        bandflux : float or `~numpy.ndarray`
            Flux in photons / s / cm^2, unless `zp` and `zpsys` are
            given, in which case flux is scaled so that it corresponds
            to the requested zeropoint. Return value is `float` if all
            input parameters are scalars, `~numpy.ndarray` otherwise.
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
        """Magnitude through the given bandpass(es), and for the given
        magnitude system(s).

        Parameters
        ----------
        band : str or list_like
            Name(s) of bandpass in registry.
        magsys : str or list_like
            Name(s) of `~sncosmo.MagSystem` in registry.

        Returns
        -------
        mag : float or `~numpy.ndarray`
            Magnitude for each item in band, magsys.
            The return value is a float if all parameters are not iterables.
            The return value is an `~numpy.ndarray` if any are iterable.
        """
        return -2.5 * np.log10(self.bandflux(band, 0., magsys))
