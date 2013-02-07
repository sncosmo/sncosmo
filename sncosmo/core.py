# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sys
import math
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from astropy.table import Table
from astropy import cosmology

__all__ = ['Bandpass', 'Spectrum']

# Constants for Spectrum
h_erg_s = 6.626068e-27  # Planck constant (erg * s)
c_AA_s = 2.9979e+18  # Speed of light ( AA / sec)

default_cosmology = cosmology.WMAP7

class Bandpass(object):
    """A bandpass, e.g., filter. ('filter' is a built-in python function.)

    Parameters
    ----------
    wavelengths : list_like
        Wavelength values, in angstroms
    transmission : list_like
        Transmission values.
    copy : bool, optional
        Copy input arrays.
    """

    def __init__(self, wavelengths, transmission):
        
        self._wavelengths = np.asarray(wavelengths)
        self._transmission = np.asarray(transmission)
        if self._wavelengths.shape != self._transmission.shape:
            raise ValueError('shape of wavelengths and transmission must match')
        if self._wavelengths.ndim != 1:
            raise ValueError('only 1-d arrays supported')

    @property
    def wavelengths(self):
        """Wavelengths in Angstroms"""
        return self._wavelengths

    @property
    def transmission(self):
        """Transmission fraction"""
        return self._transmission


class Spectrum(object):
    """A spectrum, representing wavelength and flux values.

    Parameters
    ----------
    wavelength : list_like
        Wavelength values, in angstroms
    flux : list_like
        Flux values, in units :math:`F_\lambda` (ergs/s/cm^2/Angstrom)
    fluxerr : list_like, optional
        1 standard deviation uncertainty on flux values.
    z : float, optional
        Redshift of spectrum (default is ``None``)
    dist : float, optional
        Luminosity distance in Mpc, used to adjust flux upon redshifting.
        The default is ``None``.
    meta : OrderedDict, optional
        Metadata.
    copy : bool, optional
        Copy input arrays.
    """

    def __init__(self, wavelengths, flux, fluxerr=None, z=None, dist=None,
                 meta=None):
        
        self._wavelengths = np.asarray(wavelengths)
        self._flux = np.asarray(flux)
        
        if self._wavelengths.shape != self._flux.shape:
            raise ValueError('shape of wavelength and flux must match')
        if self._wavelengths.ndim != 1:
            raise ValueError('only 1-d arrays supported')
        self._z = z
        self._dist = dist
        if fluxerr is not None:
            self._fluxerr = np.asarray(fluxerr)
            if self._wavelengths.shape != self._fluxerr.shape:
                raise ValueError('shape of wavelength and variance must match')
        else:
            self._fluxerr = None
        if meta is not None:
            self.meta = deepcopy(meta)
        else:
            self.meta = None

    @property
    def wavelengths(self):
        """Wavelengths of spectrum in Angstroms"""
        return self._wavelengths
        
    @property
    def flux(self):
        """Fluxes in ergs/s/cm^2/Angstrom"""
        return self._flux

    @property
    def fluxerr(self):
        """Fluxes in ergs/s/cm^2/Angstrom"""
        return self._fluxerr

    @property
    def z(self):
        """Redshift of spectrum."""
        return self._z

    @z.setter
    def z(self, value):
        self._z = value

    @property
    def dist(self):
        """Distance to object."""
        return self._dist

    @dist.setter
    def dist(self, value):
        self._dist = value


    def synphot(self, band):
        """Perform synthentic photometry in a given bandpass.
      
        Parameters
        ----------
        band : Bandpass object

        Returns
        -------
        flux : float
            Total flux in photons/sec/cm^2
        fluxerr : float
            Error on flux. Only returned if spectrum.fluxerr is not `None`.
        """

        # If the bandpass is not fully inside the defined region of the spectrum
        # return None.
        if (band.wavelengths[0] < self._wavelengths[0] or
            band.wavelengths[-1] > self._wavelengths[-1]):
            return None

        # Get the spectrum index range to use
        idx = ((self._wavelengths > band.wavelengths[0]) & 
               (self._wavelengths < band.wavelengths[-1]))

        # Spectrum quantities in this wavelength range
        wl = self._wavelengths[idx]
        f = self._flux[idx]
        binwidth = np.gradient(wl) # Width of each bin

        # Interpolate bandpass transmission to these wavelength values
        trans = np.interp(wl, band.wavelengths, band.transmission)

        # Convert flux from erg/s/cm^2/AA to photons/s/cm^2/AA
        factor = wl / (h_erg_s * c_AA_s)
        f *= factor

        # Get total erg/s/cm^2
        ftot = np.sum(f * trans * binwidth)

        if self._fluxerr is None:
            return ftot
        else:
            fe = self._fluxerr[idx]
            fe *= factor  # Convert from erg/s/cm^2/AA to photons/s/cm^2/AA
            fetot = np.sum((fe * trans) ** 2 * binwidth)
            return totflux, fetot


    def redshifted_to(self, z, adjust_flux=False, dist=None, cosmo=None):
        """Return a new Spectrum object at a new redshift.

        The current redshift must be defined (self.z cannot be `None`).
        A factor of (1 + z) / (1 + self.z) is applied to the wavelength. 
        The inverse factor is applied to the flux so that the bolometric
        flux remains the same.
        
        Parameters
        ----------
        z : float
            Target redshift.
        adjust_flux : bool, optional
            If True, the bolometric flux is adjusted by
            ``F_out = F_in * (D_in / D_out) ** 2``, where ``D_in`` and
            ``D_out`` are current and target luminosity distances,
            respectively. ``D_in`` is self.dist. If self.dist is ``None``,
            the distance is calculated from the current redshift and
            given cosmology.
        dist : float, optional
            Output distance in Mpc. Used to adjust bolometric flux if
            ``adjust_flux`` is ``True``. Default is ``None`` which means
            that the distance is calculated from the redshift and the
            cosmology.
        cosmo : `~astropy.cosmology.Cosmology` instance, optional
            The cosmology used to estimate distances if dist is not given.
            Default is ``None``, which results in using the default
            cosmology.

        Returns
        -------
        spec : Spectrum object
            A new spectrum object at redshift z.
        """

        if cosmo is None:
            cosmo = default_cosmology

        # Shift wavelengths, adjust flux so that bolometric flux
        # remains constant.
        factor =  (1. + z) / (1. + self._z)
        wl = self._wavelengths * factor
        f = self._flux / factor
        if self._fluxerr is not None: fe = self._fluxerr / factor
        else: fe = None

        if adjust_flux:
            # Check current distance
            if self._dist is None and self._z == 0.:
                raise ValueError("When current redshift is 0 and adjust_flux "
                                 "is requested, current distance must be "
                                 "defined")

            # Check requested distance
            if dist is None and z == 0.:
                raise ValueError("When redshift is 0 and adjust_flux "
                                 "is requested, dist must be defined")

            if self._dist is None:
                dist_in = cosmo.luminosity_distance(self._z)
            else:
                dist_in = self._dist

            if dist is None:
                dist_out = cosmo.luminosity_distance(z)
            else:
                dist_out = dist

            if dist_in <= 0. or dist_out <= 0.:
                raise ValueError("Distances must be greater than 0.")

            # Adjust the flux
            factor = (dist_in / dist_out) ** 2
            f *= factor
            if fe is not None: fe *= factor

        return Spectrum(wl, f, fluxerr=fe, z=z, dist=dist, meta=self.meta)
