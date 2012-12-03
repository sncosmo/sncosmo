# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Class to represent an astronomical survey."""

import numpy as np
from astropy.table import Table
from astropy import cosmology

from . import models

__all__ = ['Bandpass', 'Spectrum', 'Survey']

# Constants for Survey
wholesky_sqdeg = 4. * np.pi * (180. / np.pi) ** 2
drange_pad = (20., 20.)


# Constants for Spectrum

#h_erg_s = cgs.h
#c_AA_s = cgs.c * 1.e8
h_erg_s = 6.626068e-27  # Planck constant (erg * s)
c_AA_s = 2.9979e+18  # Speed of light ( AA / sec)

cosmo = cosmology.LambdaCDM(H0=70., Om0=0.3, Ode0=0.7)



class Bandpass(object):
    """A bandpass, e.g., filter. ('filter' is a built-in python function.)

    Parameters
    ----------
    wavelength : list_like
        Wavelength values, in angstroms
    transmission : list_like
        Transmission values.
    copy : bool, optional
        Copy input arrays.
    """

    def __init__(self, wavelength, transmission, copy=False):
        
        wavelength = np.array(wavelength, copy=copy)
        transmission = np.array(transmission, copy=copy)
        if wavelength.shape != transmission.shape:
            raise ValueError('shape of wavelength and transmission must match')
        if wavelength.ndim != 1:
            raise ValueError('only 1-d arrays supported')
        self.wavelength = wavelength
        self.transmission = transmission


    def size(self):
        return self.wavelength.shape[0]


    def range(self):
        return (self.wavelength[0], self.wavelength[-1])


class Spectrum(object):
    """A spectrum.

    Parameters
    ----------
    wavelength : list_like
        Wavelength values, in angstroms
    flux : list_like
        Flux values, in units f_lambda (ergs/s/cm^2/AA)
    variance : list_like, optional
        Variance on flux values.
    z : float, optional
        Redshift of spectrum (default is `None`)
    dist : float, optional
        Luminosity distance in Mpc, used to adjust flux (default is `None`)
    meta : OrderedDict, optional
        Metadata.
    copy : bool, optional
        Copy input arrays.
    """

    def __init__(self, wavelength, flux, z=None, dist=None, variance=None,
                 meta=None, copy=False):
        
        self.wavelength = np.array(wavelength, copy=copy)
        self.flux = np.array(flux, copy=copy)
        if self.wavelength.shape != self.flux.shape:
            raise ValueError('shape of wavelength and flux must match')
        if self.wavelength.ndim != 1:
            raise ValueError('only 1-d arrays supported')
        self.z = z
        self.dist = dist
        if variance is not None:
            self.variance = np.array(variance, copy=copy)
            if self.wavelength.shape != self.variance.shape:
                raise ValueError('shape of wavelength and variance must match')
        else:
            self.variance = None
        self.meta = OrderedDict() if meta is None else deepcopy(meta)


    def synphot(self, band):
        """Perform synthentic photometry in a given bandpass.
      
        Parameters
        ----------
        band : Bandpass object
            Self explanatory.

        Returns
        -------
        flux : float
            Total flux in photons/sec/cm^2
        fluxerr : float
            Error on flux, only returned if spectrum.variance is not `None`
        """

        # If the bandpass is not fully inside the defined region of the spectrum
        # return None.
        if (band.wavelength[0] < spec.wavelength[0] or
            band.wavelength[-1] > spec.wavelength[-1]):
            return None

        # Get the spectrum index range to use
        idx = ((spec.wavelength > band.wavelength[0]) & 
               (spec.wavelength < band.wavelength[-1]))

        # Spectrum quantities in this wavelength range
        wave = spec.wavelength[idx]
        flux = spec.flux[idx]
        binwidth = np.gradient(wave) # Width of each bin

        # Interpolate bandpass transmission to these wavelength values
        trans = np.interp(wave, band.wavelength, band.transmission)

        # Convert flux from erg/s/cm^2/AA to photons/s/cm^2/AA
        factor = (wave / (h_erg_s * c_AA_s))
        flux *= factor

        # Get total erg/s/cm^2
        totflux = np.sum(flux * trans * binwidth)

        if spec.variance is None:
            return totflux

        else:
            var = spec.variance[idx]
            var *= factor ** 2  # Convert from erg/s/cm^2/AA to photons/s/cm^2/AA
            totvar = np.sum(var * trans ** 2 * binwidth) # Total variance
            return totflux, np.sqrt(totvar)


    def redshifted_to(self, z, dist=None, adjust_flux=False, cosmo=None):
        """Return a new Spectrum object at a new redshift.

        The current redshift must be defined (spec.z cannot be `None`).

        Parameters
        ----------
        z : float
            A factor of (1 + z) / (1 + self.z) is applied to the wavelength. 
            The inverse factor is applied to the flux so that the bolometric
            flux remains the same.
        dist : 

        dist_in OR zdist_in : float
            Input distance in Mpc or input redshift (> 0). Used to adjust
            bolometric flux, as F_out = F_in * (D_in / D_out) ** 2,
            where D is luminosity distance. D is given by dist_in or calculated
            for the redshift zdist_in. 
        dist_out OR zdist_out :float
            Output distance in Mpc or redshift (> 0). Used to adjust bolometric
            flux. See above.

        Returns
        -------
        spec : Spectrum object
            A new spectrum object at redshift z.
        """

        # Shift wavelengths, adjust flux so that bolometric flux
        # remains constant.
        factor =  (1. + z_out) / (1. + z_in)
        new_wave = spec.wavelength * factor
        new_flux = spec.flux / factor
        if spec.variance is not None:
            new_var = spec.variance / factor ** 2
        else:
            new_var = None

        # Check flux distance inputs
        if (dist_in is not None) and (zdist_in is not None):
            raise ValueError("cannot specify both zdist_in and dist_in")
        if zdist_in is not None:
            if zdist_in <= 0.:
                raise ValueError("zdist_in must be greater than 0")
            dist_in = cosmo.luminosity_distance(zdist_in)

        # Check flux distance outputs
        if (dist_out is not None) and (zdist_out is not None):
            raise ValueError("cannot specify both zdist_out and dist_out")
        if zdist_out is not None:
            if zdist_out <= 0.:
                raise ValueError("zdist_out must be greater than 0")
            dist_out = cosmo.luminosity_distance(zdist_out)

        # Check for only one being specified.
        if (((dist_in is None) and (dist_out is not None)) or
            ((dist_in is not None) and (dist_out is None))):
            raise ValueError("Only input or only output flux distance specified")

        # If both are specified, adjust the flux.
        if dist_in is not None and dist_out is not None:
            factor = (dist_in / dist_out) ** 2
            new_flux *= factor
            if new_var is not None:
                new_var *= factor ** 2

        return Spectrum(new_wave, new_flux, variance=new_var, meta=spec.meta)




class Survey(object):
    """An astronomical transient survey.

    Parameters
    ----------
    fields: dict
        Information about the observed fields, indexed by field id (int)
    obs : astropy.table.Table, numpy.ndarray, or dict of list_like
        Table of observations in the survey. This table must have certain
        field names. See "Notes" section.
    bands : dict of Bandpass
        Dictionary of bandpasses that the survey should know about.
        The keys should be strings. In the `obs` table, `band` entries are
        strings corresponding to these keys.

    Notes
    -----
    The following data fields **must** be in the `obs` table:

        field
            integer id of observed field
        date
            Date of observations in days (e.g., MJD)
        band
            Bandpass of observation (string)
        ccdgain
            CCD gain of observations in e-/ADU
        ccdnoise
            CCD noise of observations in ADU
        skysig
            Pixel-to-pixel standard deviation in background in ADU
        psffwhm
            Full-with at half max (FWHM) of PSF in pixels.
        zp
            Zeropoint of observations (float).
        zpsys
            Zeropoint system (string).

    The following are **optional** fields in the `obs` table
    (not yet implemented - for now these are ignored).

        psf2
            TODO description
        zperr
            Systematic uncertainty in zeropoint.
    """

    def __init__(self, fields, obs):
        self.fields = fields
        self.obs = Table(obs)
        
        # TODO: check that obs includes all necessary data fields


        # Check that

    def simulate(self, tmodel, params={}, vrate=1.e-4, cosmo=None,
                 z_range=(0., 2.), z_bins=40):
        """Run a simulation of the survey.
        
        Parameters
        ----------
        tmodel : A TransientModel instance
            The transient we're interested in.
        params : dictionary or callable, optional
            Dictionary of parameters to pass to the model or
            a callable that returns such a dictionary on each call. 
            Typically the callable would randomly select parameters
            from some underlying distribution on each call. The default is
            no parameters.
        vrate : float or callable, optional
            The volumetric rate in (comoving Mpc)^{-3} yr^{-1} or
            a callable that returns the rate as a function of redshift.
            (The default is 1.e-4.)
        cosmo : astropy.cosmology.Cosmology, optional
            Callable returning a dictionary of parameters (randomly selected
            from some underlying distribution) to pass to the model. 
            (The default is `None`, which implies the WMAP7 cosmology.)
        z_range : (float, float), optional
            Redshift range in which to generate transients.
            The default is (0., 2.).
        z_bins : float, optional
            Number of redshift bins (the default is 40).
        """

        # Check the transient model.
        if not isinstance(tmodel, models.TransientModel):
            raise ValueError('tmodel must be a TransientModel instance')

        # Check the model parameters.
        if params is None:
            params = {}

        # Check the volumetric rate.
        if not callable(vrate):
            vrate = lambda z: float(vrate)

        # Check the cosmology.
        if cosmo is None:
            cosmo = cosmology.WMAP7
        elif not isinstance(cosmo, cosmology.Cosmology):
            raise ValueError('cosmo must be a Cosmology instance')

        # Check the redshifts
        if len(z_range) != 2:
            raise ValueError('z_range must be length 2')
        z_min, z_max = z_range
        if not (z_max > z_min):
            raise ValueError('z_max must be greater than z_min')
        z_bins = int(z_bins)
        if z_bins < 1:
            raise ValueError('z_bins must be greater than 0')

        # Get volumes in each redshift shell over whole sky
        z_binedges = np.linspace(z_min, z_max, z_bins + 1) 
        sphere_vols = cosmo.comoving_volume(z_binedges) 
        shell_vols = sphere_volumes[1:] - sphere_volumes[:-1]

        # Get list of unique bandpasses used in survey
        bands = np.unique(self.obs['band'])

        # Load bandpasses.
        bandpasses = {}
        for band in bands:
            bandpasses[band] = Bandpass.get(band)

        # Get list of unique field id's
        fids = np.unique(self.obs['field'])

        # Loop over fields
        for fid in fids:

            # Observations in just this field
            fobs = self.obs(self.obs['field'] == fid)
            
            # Get range of observation dates.
            drange = (fobs['date'].min(), fobs['date'].max())

            # Loop over redshift bins in this field
            for z_lo, z_hi, shell_vol in zip(z_binedges[:-1],
                                             z_binedges[1:],
                                             shell_vols):
                z_mid = (z_lo + z_hi) / 2.
                bin_vol = shell_vol * area / wholesky_sqdeg
                simdrange = (drange[0] - drange_pad[0] * (1 + z_mid),
                             drange[1] + drange_pad[1] * (1 + z_mid))
                time_rframe = (simdrange[1] - simdrange[0]) / (1 + z_mid)

                # Number of transients in this bin
                intrinsic_rate = vrate(z_mid) * bin_vol * time_rframe
                ntrans = np.random.poisson(intrinsic_rate, 1)[0]
                if ntrans == 0: continue

                # Where are they in time and redshift?
                dates = np.random.uniform(simdrange[0], simdrange[1], ntrans)
                zs = np.random.uniform(z_lo, z_hi, ntrans)

                # Loop over the transients that occured in this bin
                for i in range(ntrans):
                    
                    # Get randomly selected parameters for this transient
                    params = param_gen()
                    transient = {'date_max': dates[i], 'z': zs[i],
                                 'date': [], 'mag': [],
                                 'flux': [], 'fluxerr':[]}
                    

                    # Get mag, flux, fluxerr in each observation in `fobs`
                    for j in range(len(fobs)):
                        
                        phase = (fobs[j]['date'] - date_max) / (z + 1)
                        spec = Spectrum(tmodel.wavelengths(),
                                        tmodel.flux(phase, **params))
                        
                        # redshift the spectrum
                        # do synthetic photometry
                        # save results to dict
