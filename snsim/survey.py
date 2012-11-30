# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Class to represent an astronomical survey."""

import numpy as np
from astropy.table import Table
from astropy import cosmology

from .bandpass import Bandpass
from .spectrum import Spectrum
from . import models

# Constants defined for convenience
wholesky_sqdeg = 4. * np.pi * (180. / np.pi) ** 2

drange_pad = (20., 20.)

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
