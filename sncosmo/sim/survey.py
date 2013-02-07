# Tools for simulation of transients.

import sys
import math
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from astropy.table import Table
from astropy import cosmology

__all__ = ['Survey']

# Constants for Survey
wholesky_sqdeg = 4. * np.pi * (180. / np.pi) ** 2

class Survey(object):
    """An astronomical transient survey.

    Parameters
    ----------
    fields : dict
        Information about the observed fields, indexed by field id (int)
    obs : astropy.table.Table, numpy.ndarray, or dict of list_like
        Table of observations in the survey. This table must have certain
        field names. See "Notes" section.
    bandpasses : dict of Bandpass
        Dictionary of bandpasses that the survey should know about.
        The keys should be strings. In the ``obs`` table, ``'band'`` entries are
        strings corresponding to these keys.
    zpspectra : dict of Spectrum
        Dictionary of zeropoint spectra. The keys should be strings. In
        the ``obs`` table, the ``'zpsys'`` field corresponds to these keys.

    Notes
    -----

    The following data fields are recognized in the `obs` table:

    * ``'field'``: Integer id of observed field **(required)**
    * ``'date'``: Date of observations (e.g., MJD) [days]
    * ``'band'``: Bandpass of observation (string)
    * ``'ccdgain'``: CCD gain of observations in [e-/ADU]
    * ``'ccdnoise'``: CCD noise of observations [ADU]
    * ``'skyskig'``: Pixel-to-pixel standard deviation in background. [ADU]
    * ``'psfsig'``: 1-sigma width of gaussian PSF. [pixels]
    * ``'zp'``: Zeropoint of observations (float).
    * ``'zpsys'``: Zeropoint system (string).
    * ``'psf2'``: No description **(optional)**
    * ``'zperr'``: Systematic uncertainty in zeropoint. **(optional)**

    The following data fields are recognized in the "fields" dictionary:

    * ``'area'``: Area in square degrees. **(required)** 

    """

    def __init__(self, fields, obs, bandpasses, zpspectra):
        self.fields = fields
        self.obs = Table(obs)
        self.bandpasses = bandpasses
        self.zpspectra = zpspectra
        
        # Check that required keys are in the observation table
        required_keys = ['field', 'date', 'band', 'ccdgain', 'ccdnoise',
                         'skysig', 'psfsig', 'zp', 'zpsys']
        for key in required_keys:
            if not key in self.obs.colnames:
                raise ValueError("observations missing required key: '{}'"
                                 .format(key))

        # Check that observed bands are in self.bandpasses
        uniquebands = np.unique(self.obs['band'])
        for band in uniquebands:
            if not band in self.bandpasses:
                raise ValueError("Bandpass '{}' is in observations, but not"
                                 " in 'bandpasses' dictionary."
                                 .format(band))

        # Check that zeropoint systems are in self.zpspectra
        for name in self.obs['zpsys']:
            if not name in self.zpspectra:
                raise ValueError("zeropoint system '{}' is in observations, "
                                 "but not in 'zpspectra' dictionary."
                                 .format(name))

        # get the zp synthetic flux for all bandpass, zpsys combinations.
        self._zpflux = {}
        for bandname, bandpass in self.bandpasses.iteritems():
            for zpsys, zpspec in self.zpspectra.iteritems():
                self._zpflux[(bandname, zpsys)] = zpspec.synphot(bandpass)


    def simulate(self, tmodel, params, mband, zpsys, vrate=1.e-4, cosmo=None,
                 z_range=(0., 2.), z_bins=40, verbose=False):
        """Run a simulation of the survey.
        
        Parameters
        ----------
        tmodel : A models.Transient instance
            The transient we're interested in.
        params : dictionary or callable
            Dictionary of parameters to pass to the model *or*
            a callable that returns such a dictionary on each call. 
            Typically the callable would randomly select parameters
            from some underlying distribution on each call.
            The parameters must include 'm', the absolute magnitude.
        mband : str
            The rest-frame bandpass in which the absolute magnitude is 
            measured. Must be in the Survey's ``bandpasses`` dictionary.
        zpsys : str
            The zeropoint system of the absolute magnitude ``m``. Must be
            in the Survey's ``zpspectra`` dictionary.
        vrate : float or callable, optional
            The volumetric rate in (comoving Mpc)^{-3} yr^{-1} or
            a callable that returns the rate as a function of redshift.
            (The default is 1.e-4.)
        cosmo : astropy.cosmology.Cosmology, optional
            Cosmology used to determine volumes and luminosity distances.
            (The default is `None`, which implies the WMAP7 cosmology.)
        z_range : (float, float), optional
            Redshift range in which to generate transients.
            The default is (0., 2.).
        z_bins : float, optional
            Number of redshift bins (the default is 40).
        """

        # Check the transient model.
        if not isinstance(tmodel, models.Transient):
            raise ValueError('tmodel must be a sncosmo.models.Transient '
                             'instance')

        # Make params a callable if it isn't already.
        if callable(params):
            getparams = params
        else:
            constant_params = deepcopy(params)
            def getparams(): return constant_params

        # Check that 'm' is in the dictionary that getparams() returns.
        if 'm' not in getparams():
            raise ValueError("params must include 'm'")

        # Check that mband is in bandpasses
        if not mband in self.bandpasses:
            raise ValueError("Requested 'mband' {} not in survey bandpasses."
                             .format(mband))

        # Check that zpsys is in the survey's zpspectra
        if not zpsys in self.zpspectra:
            raise ValueError("Requested 'zpsys' {} not in survey zpspectra."
                             .format(zpsys))

        # Check the volumetric rate.
        if not callable(vrate):
            constant_vrate = float(vrate)
            vrate = lambda z: constant_vrate

        # Check the cosmology.
        if cosmo is None:
            cosmo = default_cosmology
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

        # Initialize output transients (list)
        transients = []

        # Get volumes in each redshift shell over whole sky
        z_binedges = np.linspace(z_min, z_max, z_bins + 1) 
        sphere_vols = cosmo.comoving_volume(z_binedges) 
        shell_vols = sphere_vols[1:] - sphere_vols[:-1]

        fids = np.unique(self.obs['field'])  # Unique field IDs.

        # Loop over redshift bins.
        if verbose: print "Redshift Range     Transient"
        for z_lo, z_hi, shell_vol in zip(z_binedges[:-1], z_binedges[1:],
                                         shell_vols):
            z_mid = (z_lo + z_hi) / 2.

            ntrans_bin = 0
            if verbose:
                print "{:5.3f} < z < {:5.3f} {:5d}".format(z_lo, z_hi, 0),
                sys.stdout.flush()

            # Loop over fields
            for fid in fids:

                # Observations in just this field
                fobs = self.obs[self.obs['field'] == fid]
            
                # Get range of observation dates for this field
                drange = (fobs['date'].min(), fobs['date'].max())

                bin_vol = shell_vol * self.fields[fid]['area'] / wholesky_sqdeg

                # Simulate transients in a wider date range than the
                # observations. This is the range of dates that phase=0 
                # will be placed at.
                simdrange = (drange[0] - tmodel.phases()[-1] * (1 + z_mid),
                             drange[1] - tmodel.phases()[0] * (1 + z_mid))
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
                    
                    date0 = dates[i] # date corresponding to phase = 0
                    z = zs[i]  # redshift of transient

                    # Get a random selection of model parameters.
                    # Do a deep copy in case getparams() returns the same
                    # object each time (we will modify our copy below).
                    params = deepcopy(getparams())
                    absmag = params.pop('m') # Get absolute mag out of it.
                    

                    # Normalize the spectral surface based on `absmag`.
                    # Spectrum at phase = 0:
                    spec = Spectrum(tmodel.wavelengths(),
                                    tmodel.flux(0., **params),
                                    z=0., dist=1.e-5)

                    # Do synthetic photometry in rest-frame normalizing band.
                    flux = spec.synphot(self.bandpasses[mband])
                    zpflux = self._zpflux[(mband, zpsys)]

                    # Amount to add to mag to make it the target rest-frame
                    # absolute magnitude.
                    magdiff = absmag + 2.5 * math.log10(flux / zpflux)

                    # Initialize a data table for this transient
                    transient_meta = {'date0': date0, 'z': z}
                    for p, val in params.iteritems(): transient_meta[p] = val
                    transient_data = {'date': [], 'band': [], 'mag': [],
                                      'flux': [], 'fluxerr':[]}
                    
                    # Get mag, flux, fluxerr in each observation in `fobs`
                    for j in range(len(fobs)):
                        
                        phase = (fobs[j]['date'] - date0) / (z + 1)
                        if (phase < tmodel.phases()[0] or
                            phase > tmodel.phases()[-1]):
                            continue

                        # Get model spectrum for selected phase and parameters.
                        spec = Spectrum(tmodel.wavelengths(),
                                        tmodel.flux(phase, **params),
                                        z=0., dist=1.e-5)

                        # Redshift it.
                        spec = spec.redshifted_to(z, adjust_flux=True,
                                                  cosmo=cosmo)

                        # Get mag in observed band.
                        flux = spec.synphot(self.bandpasses[fobs[j]['band']])
                        if flux is None: continue
                        zpflux = self._zpflux[(fobs[j]['band'],
                                               fobs[j]['zpsys'])]
                        mag = magdiff - 2.5 * math.log10(flux / zpflux)
                        
                        # ADU on the image.
                        flux = 10 ** (0.4 * (fobs[j]['zp'] - mag))

                        # calculate uncertainty in flux based on ccdgain,
                        # ccdnoise, skysig, psfsig.
                        noise_area = 4. * math.pi * fobs[j]['psfsig']
                        bkgpixnoise = fobs[j]['ccdnoise'] + fobs[j]['skysig']
                        fluxerr = math.sqrt((noise_area * bkgpixnoise) ** 2 +
                                            flux / fobs[j]['ccdgain'])

                        # Scatter fluxes by the fluxerr
                        flux = np.random.normal(flux, fluxerr)

                        transient_data['date'].append(fobs[j]['date'])
                        transient_data['band'].append(fobs[j]['band'])
                        transient_data['mag'].append(mag)
                        transient_data['flux'].append(flux)
                        transient_data['fluxerr'].append(fluxerr)
                        
                    # save transient.
                    transients.append(Table(transient_data, 
                                            meta=transient_meta))
                    ntrans_bin += 1
                    if verbose:
                        print 6 * "\b" + "{:5d}".format(ntrans_bin),
                        sys.stdout.flush()

            if verbose: print ""
                
        return transients
