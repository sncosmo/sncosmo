# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Class to represent an astronomical survey."""

import numpy as np
from astropy import table
from astropy.io import ascii
from astropy import cosmology

from .bandpass import Bandpass
from .spectrum import Spectrum

cosmo = cosmology.LambdaCDM(H0=70., Om0=0.3, Ode0=0.7)

sqdeg_per_steradian = (180. / np.pi) ** 2
wholesky_sqdeg = 4. * np.pi * sqdeg_per_steradian

drange_pad = (20., 20.)

class Survey(object):
    """An astronomical transient survey.

    Parameters
    ----------
    filename : str, optional
        File from which to read survey information. If given, observation
        parameters are read from file and other keywords are ignored.
    observations : dict or numpy.ndarray
        Parameters describing observations in dictionary or structured ndarray.
        Recognized fields are:

        fieldid
            integer id of observation field
        date
            Date of observations in days (e.g., MJD)
        band
            Bandpass of observation
        ccdgain
            CCD gain of observations (e-/ADU)
        ccdnoise
            CCD noise of observations in ADU
        skysig
            Pixel-to-pixel standard deviation in background in ADU
        seeing
            Seeing in arcseconds.
        pixelscale
            Pixel scale in arcseconds.
        zp
            Zeropoint of observations
        zpsys
            Zeropoint system.
        zpsyserr
            Systematic uncertainty in zeropoint.
    """

    def __init__(self, **kwargs):
        if 'filename' in kwargs and kwargs['filename'] is not None:
            self._init_from_file(kwargs['filename'])
        else:
            self._init_from_dict(kwargs)
        

    def _init_from_file(self, filename):
        """Initialize from a text file."""
        t = ascii.read(filename, guess=False)
        self._obs = table._data


    def _init_from_dict(self, d):
        """Initialize from a dictionary of parameters."""
        t = table.Table(d)
        self._obs = t._data

    def __str__(self):
        return str(table.Table(self._obs))

    def simulate(self, tmodel, vrate, param_gen,
                 z_min=0., z_max=2., z_step=0.05):
        """Run a simulation of the survey.
        
        Parameters
        ----------
        tmodel : A TransientModel instance
            The transient we're interested in.
        vrate : callable
            Function giving the volumetric rate in (comoving Mpc)^{-3} yr^{-1}
            as a function of redshift.
        param_gen : callable
            Callable returning a dictionary of parameters (randomly selected
            from some underlying distribution) to pass to the model.
        z_min : float
            Lowest redshift to generate transients at
        z_max : float
            Highest redshift to generate transients at.
        z_step : float
            Redshift step
        """

        # Temporary stuff
        area = 1.
        def trate(z):
            if z < 1:
                return 0.25e-4 * (1 + 2.5 * z)
            else:
                return 0.25e-4 * 3.5

        # Check tmodel.

        # Get volumes in each redshift shell over whole sky
        z_bins = np.arange(z_min, z_max, z_step) 
        sphere_vols = cosmo.comoving_volume(z_bins) 
        shell_vols = sphere_volumes[1:] - sphere_volumes[:-1]

        # Get list of unique bandpasses used in survey
        bands = np.unique(self._obs['band'])

        # Load bandpasses.
        bandpasses = {}
        for band in bands:
            bandpasses[band] = Bandpass.get(band)

        # Get list of unique field id's
        fieldids = np.unique(self._obs['fieldid'])

        # Loop over fields
        for fid in fieldids:
            fobs = self._obs(self._obs == fid)  # observations for this field
            
            # Get range of observation dates.
            drange = (fobs['date'].min(), fobs['date'].max())

            # Loop over redshift bins in this field
            for z_lo, z_hi, shell_vol in zip(z_bins[:-1], z_bins[1:],
                                             shell_vols):
                z_mid = (z_lo + z_hi) / 2.
                bin_vol = shell_vol * area / wholesky_sqdeg
                simdrange = (drange[0] - drange_pad[0] * (1 + z_mid),
                             drange[1] + drange_pad[1] * (1 + z_mid))
                time_rframe = (simdrange[1] - simdrange[0]) / (1 + z_mid)

                # Number of transients in this bin
                intrinsic_rate = trate(z_mid) * bin_vol * time_rframe
                ntrans = np.random.poisson(intrinsic_rate , 1)[0]
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
