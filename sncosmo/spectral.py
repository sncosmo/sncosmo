# Licensed under a 3-clause BSD style license - see LICENSE.rst

from copy import deepcopy
from collections import OrderedDict

import numpy as np
from astropy import units as u
from astropy import cosmology
import astropy.constants as const

from . import registry

__all__ = ['Bandpass', 'Spectrum', 'MagnitudeSystem', 'ABSystem']

class Bandpass(object):
    """Transmission as a function of spectral dispersion.

    Parameters
    ----------
    dispersion : list_like
        Dispersion. Monotonically increasing values.
    transmission : list_like
        Transmission fraction.
    dispersion_unit : `~astropy.units.Unit` or str
        Dispersion unit. Default is Angstroms.
    name : str
        Identifier.
    """

    def __init__(self, dispersion, transmission, dispersion_unit=u.AA,
                 meta=None):
        
        self._dispersion = np.asarray(dispersion) 
        self._transmission = np.asarray(transmission)
        if self._dispersion.shape != self._transmission.shape:
            raise ValueError('shape of wavelengths and '
                             'transmission must match')
        if self._dispersion.ndim != 1:
            raise ValueError('only 1-d arrays supported')
        self._dunit = dispersion_unit

        if meta is None:
            self.meta = OrderedDict()
        else:
            self.meta = copy.deepcopy(meta)

    @property
    def dispersion(self):
        """Dispersion values (always monotonically increasing)."""
        return self._dispersion

    @property
    def transmission(self):
        """Transmission values corresponding to dispersion values."""
        return self._transmission

    @property
    def dunit(self):
        """Dispersion unit"""
        return self._dunit

    def to_unit(new_unit):
        """Return a new bandpass instance with the requested dispersion units.
        
        A new instance is necessary because the dispersion and transmission
        values may need to be reordered to ensure that they remain
        monotonically increasing.

        If the requested units are the same as the current units, self is
        returned.
        """

        if dunit == self._dunit:
            return self
        d = self._dunit.to(new_unit, self._dispersion, u.spectral())
        t = self._transmission
        if d[0] > d[-1]:
            d = np.flipud(d)
            t = np.flipud(t)
        return Bandpass(d, t, dispersion_unit=new_unit, meta=self.meta)

    def __repr__(self):
        
        if 'name' in self.meta: name=self.meta['name']
        else: name = 'unnamed'
        unitstr = self._dunit.to_string()
        return ("<Bandpass '{0:s}'\n"
                "Dispersion range: {1:.6g}-{2:.6g} {3:s}>\n"
                .format(name, self._dispersion[0],
                        self._dispersion[-1], unitstr))

    @classmethod
    def from_name(cls, name):
        """Get a bandpass from the registry by name."""
        return registry.retrieve(cls, name)


class Spectrum(object):
    """A spectrum, representing dispersion and spectral density values.

    Parameters
    ----------
    dispersion : list_like
        Dispersion values.
    fluxdensity : list_like
        Flux values.
    error : list_like, optional
        1 standard deviation uncertainty on flux density values.
    dispersion_unit : `~astropy.units.Unit`
        Units 
    units : `~astropy.units.BaseUnit`
        For now, only units with flux density in energy (not photon counts).
    z : float, optional
        Redshift of spectrum (default is `None`)
    dist : float, optional
        Luminosity distance in Mpc, used to adjust flux upon redshifting.
        The default is ``None``.
    meta : OrderedDict, optional
        Metadata.
    copy : bool, optional
        Copy input arrays.
    """

    def __init__(self, dispersion, fluxdensity, error=None,
                 unit=(u.erg / u.s / u.cm**2 / u.AA), dispersion_unit=u.AA,
                 z=None, dist=None, meta=None):
        
        self._dispersion = np.asarray(dispersion)
        self._fluxdensity = np.asarray(fluxdensity)
        self._dunit = dispersion_unit
        self._unit = unit
        self._z = z
        self._dist = dist

        if error is not None:
            self._error = np.asarray(error)
            if self._dispersion.shape != self._error.shape:
                raise ValueError('shape of wavelength and variance must match')
        else:
            self._error = None

        if meta is None:
            self.meta = OrderedDict()
        else:
            self.meta = copy.deepcopy(meta)

        if self._dispersion.shape != self._fluxdensity.shape:
            raise ValueError('shape of wavelength and fluxdensity must match')
        if self._dispersion.ndim != 1:
            raise ValueError('only 1-d arrays supported')


    @property
    def dispersion(self):
        """Dispersion values."""
        return self._dispersion
        
    @property
    def fluxdensity(self):
        """Spectral flux density values"""
        return self._fluxdensity

    @property
    def error(self):
        """Uncertainty on flux density."""
        return self._error

    @property
    def dunit(self):
        """Units of dispersion."""
        return self._dunit
        
    @property
    def unit(self):
        """Units of flux density."""
        return self._unit

    @property
    def z(self):
        """Redshift of spectrum."""
        return self._z

    @z.setter
    def z(self, value):
        self._z = value

    @property
    def dist(self):
        """Distance to object in Mpc."""
        return self._dist

    @dist.setter
    def dist(self, value):
        self._dist = value


    def flux(self, band):
        """Perform synthentic photometry in a given bandpass.
      
        The bandpass transmission is interpolated onto the dispersion grid
        of the spectrum. The result is a weighted sum of the spectral flux
        density values (weighted by transmission values).
        
        Parameters
        ----------
        band : Bandpass object or name of registered bandpass.

        Returns
        -------
        flux : float
            Total flux in ph/s/cm^2. If part of bandpass falls
            outside the spectrum, `None` is returned instead.
        fluxerr : float
            Error on flux. Only returned if the `error` attribute is not
            `None`.
        """

        band = Bandpass.from_name(band)
        band = band.to_unit(self._dunit)

        if (band.dispersion[0] < self._dispersion[0] or
            band.dispersion[-1] > self._dispersion[-1]):
            return None

        i0, i1 = np.flatnonzero((self._dispersion>band.dispersion[0]) & 
                                (self._dispersion<band.dispersion[-1]))[0, -1]
        d = self._dispersion[i0:i1]
        f = self._fluxdensity[i0:i1]

        #TODO: use spectral density equivalencies once they can do photons.
        # first convert to ergs / s /cm^2 / (dispersion unit)
        target_unit = u.erg / u.s / u.cm**2 / self._dunit
        if self._unit != target_unit:
            f = self._unit.to(target_unit, f, 
                              u.spectral_density(self._dunit, d))
        # Then convert ergs to photons: photons = Energy / (h * nu)
        f = f / const.h.cgs.value / self._dunit.to(u.Hz, d, u.spectral())

        trans = np.interp(d, band.dispersion, band.transmission)
        binw = np.gradient(d)
        ftot = np.sum(f * trans * binw)

        if self._error is None:
            return ftot

        else:
            e = self._error[i0:i1]

            # Do the same conversion as above
            if self._unit != target_unit:
                e = self._unit.to(target_unit, e, 
                                  u.spectral_density(self._dunit, d))
            e = e / const.h.cgs.value / self._dunit.to(u.Hz, d, u.spectral())
            etot = np.sqrt(np.sum(e * e * trans * binwidth))
            return ftot, etot



    def redshifted_to(self, z, adjust_flux=False, dist=None, cosmo=None):
        """Return a new Spectrum object at a new redshift.

        The current redshift must be defined (self.z cannot be `None`).
        A factor of (1 + z) / (1 + self.z) is applied to the wavelength. 
        The inverse factor is applied to the flux so that the bolometric
        flux (e.g., erg/s/cm^2) remains constant.
        
        .. note:: Currently this only works for units in ergs.

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

        if self._z is None:
            raise ValueError('Must set current redshift in order to redshift'
                             ' spectrum')

        if self._dunit.physical_type == u.m.physical_type:
            factor = (1. + z) / (1. + self._z)
        elif self._dunit.physical_type == u.Hz.physical_type:
            factor = (1. + self._z) / (1. + z)
        else:
            raise ValueError('dispersion must be in wavelength or frequency')

        d = self._dispersion * factor
        f = self._fluxdensity / factor
        if self._error is not None:
            e = self._error / factor
        else:
            e = None

        if adjust_flux:
            if self._dist is None and self._z == 0.:
                raise ValueError("When current redshift is 0 and adjust_flux "
                                 "is requested, current distance must be "
                                 "defined")
            if dist is None and z == 0.:
                raise ValueError("When redshift is 0 and adjust_flux "
                                 "is requested, dist must be defined")
            if cosmo is None:
                cosmo = cosmology.get_current()

            if self._dist is None:
                dist_in = cosmo.luminosity_distance(self._z)
            else:
                dist_in = self._dist

            if dist is None: dist = cosmo.luminosity_distance(z)

            if dist_in <= 0. or dist <= 0.:
                raise ValueError("Distances must be greater than 0.")

            # Adjust the flux
            factor = (dist_in / dist) ** 2
            f *= factor
            if e is not None: e *= factor

        return Spectrum(d, f, error=e, z=z, dist=dist, meta=self.meta,
                        unit=self._unit, dispersion_unit=self._dispersion_unit)


class MagnitudeSystem(object):
    """A class for defining a translation between magnitudes and physical
    flux values. 

    Parameters
    ----------
    refspectrum : `Spectrum`
        Reference spectrum.

    Examples
    --------
    >>> vegasys = MagnitudeSystem(vega_spec)  # `vega_spec` is a Spectrum
    >>> vegasys.zpflux('desu')  # photons/s/cm^2 through given bandpass
    >>> 

    """
    def __init__(self, refspectrum):

        self._refspectrum = refspectrum
        self._zpflux = {}

    def zpflux(self, band):
        """Return flux of an object with magnitude zero in the given bandpass.

        Parameters
        ----------
        bandpass : `~sncosmo.spectral.Bandpass` or str

        Returns
        -------
        flux : float
            Flux in some units.
        """
        band = Bandpass.from_name(band)
        try:
            return self._zpflux[band]
        except KeyError:
            pass

        self._zpflux[band] = self._refspectrum.flux(band)
        return self._zpflux[band]


class ABSystem(MagnitudeSystem):
    
    def __init__(self):
        self._zpflux = {}

    def zpflux(self, band):
        """Return flux of an object with magnitude zero in the given bandpass.

        Parameters
        ----------
        bandpass : `~sncosmo.spectral.Bandpass` or str

        Returns
        -------
        flux : float
            Flux in ph/s/cm^2

        Examples
        --------
        >>> ab = ABSystem()
        >>> f = ab.zpflux('desg')

        """
        band = Bandpass.from_name(band)
        try:
            return self._zpflux[band]
        except KeyError:
            pass
        b = band.to_unit(u.Hz)
        f = 3631. / const.h.cgs.value / b.dispersion  # convert to photons
        binw = np.gradient(b.dispersion)
        self._zpflux[band] = np.sum(f * b.transmission * binw)
        return self._zpflux[band]    


#class LocalMagSystem(MagnitudeSystem):
#    """A "local magnitude system" is defined by an absolute flux standard
#    and the magnitude of that standard in one or more bandpasses. The
#    magnitude system is therefore only defined for the given bandpasses.

#    Examples
#    --------
#    >>> chft = LocalMagSystem(bd17spectrum,
#    ...                       {'megacamu': 9.7688, 'megacamg': 9.6906,
#    ...                        'megacamr': 9.2183, 'megacami': 8.9142})
#    >>> chft.zp_cflux('megacamu') # flux an object 9.7688 mag brighter than bd17
#    >>> chft.zp_cflux('desg') # Exception
#    """
    

#class ABSystem(MagnitudeSystem):
#    """Magnitude system where a source with F_nu = 3631 Jansky at all
#    frequencies has magnitude 0 in all bands."""


