"""Tools for simulation of transients."""

import sys
import math
from copy import deepcopy

import numpy as np
from numpy import random
from scipy.interpolate import InterpolatedUnivariateSpline as Spline1d
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.utils import OrderedDict as odict

__all__ = ['zdist', 'realize_lcs', 'simulate_vol']

WHOLESKY_SQDEG = 4. * np.pi * (180. / np.pi) ** 2


def zdist(zmin, zmax, time=365.25, area=1.,
          ratefunc=lambda z: 1.e-4,
          cosmo=FlatLambdaCDM(H0=70.0, Om0=0.3)):
    """Generate a distribution of redshifts.

    Generates the correct redshift distribution and number of SNe, given
    the input volumetric SN rate, the cosmology, and the observed area and
    time.

    Parameters
    ----------
    zmin, zmax : float
        Minimum and maximum redshift.
    time : float, optional
        Time in days (default is 1 year).
    area : float, optional
        Area in square degrees (default is 1 square degree). ``time`` and
        ``area`` are only used to determine the total number of SNe to
        generate.
    ratefunc : callable
        A callable that accepts a single float (redshift) and returns the
        comoving volumetric rate at each redshift in units of yr^-1 Mpc^-3.
        The default is a function that returns ``1.e-4``.
    cosmo : `~astropy.cosmology.Cosmology`, optional
        Cosmology used to determine volume. The default is a FlatLambdaCDM
        cosmology with ``Om0=0.3``, ``H0=70.0``.

    Examples
    --------

    Loop over the generator:

    >>> for z in zdist(0.0, 0.25):  # doctest: +SKIP
    ...     print z                 # doctest: +SKIP
    ...
    0.151285827576
    0.204078030595
    0.201009196731
    0.181635472172
    0.17896188781
    0.226561237264
    0.192747368762

    This tells us that in one observer-frame year, over 1 square
    degree, 7 SNe occured at redshifts below 0.35 (given the default
    volumetric SN rate of 10^-4 SNe yr^-1 Mpc^-3). The exact number is
    drawn from a Poisson distribution.

    Generate the full list of redshifts immediately:

    >>> zlist = list(zdist(0., 0.25))

    Define a custom volumetric rate:

    >>> def snrate(z):
    ...     return 0.5e-4 * (1. + z)
    ...
    >>> zlist = list(zdist(0., 0.25, ratefunc=snrate))

    """

    # Get comoving volume in each redshift shell.
    z_bins = 100  # Good enough for now.
    z_binedges = np.linspace(zmin, zmax, z_bins + 1)
    z_binctrs = 0.5 * (z_binedges[1:] + z_binedges[:-1])
    sphere_vols = cosmo.comoving_volume(z_binedges).value
    shell_vols = sphere_vols[1:] - sphere_vols[:-1]

    # SN / (observer year) in shell
    shell_snrate = np.array([shell_vols[i] *
                             ratefunc(z_binctrs[i]) / (1.+z_binctrs[i])
                             for i in xrange(z_bins)])

    # SN / (observer year) within z_binedges
    vol_snrate = np.zeros_like(z_binedges)
    vol_snrate[1:] = np.add.accumulate(shell_snrate)

    # Create a ppf (inverse cdf). We'll use this later to get
    # a random SN redshift from the distribution.
    snrate_cdf = vol_snrate / vol_snrate[-1]
    snrate_ppf = Spline1d(snrate_cdf, z_binedges, k=1)

    # Total numbe of SNe to simulate.
    nsim = vol_snrate[-1] * (time/365.25) * (area/WHOLESKY_SQDEG)

    for i in xrange(random.poisson(nsim)):
        yield float(snrate_ppf(random.random()))


def realize_lcs(observations, model, params, thresh=None,
                trim_observations=False):
    """Realize data for a set of SNe given a set of observations.

    Parameters
    ----------
    observations : `~astropy.table.Table` or `~numpy.ndarray`
        Table of observations. Must contain the following column names:
        ``band``, ``time``, ``zp``, ``zpsys``, ``gain``, ``skynoise``.
    model : `sncosmo.Model`
        The model to use in the simulation.
    params : list (or generator) of dict
        List of parameters to feed to the model for realizing each light curve.
    thresh : float, optional
        If given, light curves are skipped (not returned) if none of the data
        points have signal-to-noise greater than ``thresh``.
    trim_observations : bool, optional
        If True, only observations with times between
        ``model.mintime()`` and ``model.maxtime()`` are included in
        result table for each SN. Default is False.

    Returns
    -------
    sne : list of `~astropy.table.Table`
        Table of realized data for each item in ``params``.

    Notes
    -----
    ``skynoise`` is the image background contribution to the flux measurement
    error (in units corresponding to the specified zeropoint and zeropoint
    system). To get the error on a given measurement, ``skynoise`` is added
    in quadrature to the photon noise from the source.

    It is left up to the user to calculate ``skynoise`` as they see fit as the
    details depend on how photometry is done and possibly how the PSF is
    is modeled. As a simple example, assuming a Gaussian PSF, and perfect
    PSF photometry, ``skynoise`` would be ``4 * pi * sigma_PSF * sigma_pixel``
    where ``sigma_PSF`` is the standard deviation of the PSF in pixels and
    ``sigma_pixel`` is the background noise in a single pixel in counts.

    """

    lcs = []

    # TODO: copy model so we don't mess up the user's model?

    # get underlying numpy structured array (if astropy.Table)
    observations = np.asarray(observations)

    for p in params:
        model.set(**p)

        # Select times for output that fall within tmin amd tmax of the model
        if trim_observations:
            mask = ((observations['time'] > model.mintime()) &
                    (observations['time'] < model.maxtime()))
            snobs = observations[mask]
        else:
            snobs = observations

        flux = model.bandflux(snobs['band'],
                              snobs['time'],
                              zp=snobs['zp'],
                              zpsys=snobs['zpsys'])

        fluxerr = np.sqrt(snobs['skynoise']**2 +
                          np.abs(flux) / snobs['gain'])

        # Scatter fluxes by the fluxerr
        flux = np.random.normal(flux, fluxerr)

        # Check if any of the fluxes are significant
        if thresh is not None and not np.any(flux/fluxerr > thresh):
            continue

        data = odict([('time', snobs['time']),
                      ('band', snobs['band']),
                      ('flux', flux),
                      ('fluxerr', fluxerr),
                      ('zp', snobs['zp']),
                      ('zpsys', snobs['zpsys'])])
        lcs.append(Table(data, meta=p))

    return lcs


def simulate_vol(obs_sets, model, gen_params, vrate,
                 cosmo=FlatLambdaCDM(H0=70., Om0=0.3),
                 z_range=(0., 1.), default_area=1., nsim=None,
                 nret=10, thresh=5.):
    """Simulate transient photometric data according to observations
    (EXPERIMENTAL).

    .. warning::

       This function is experimental in v0.4

    Parameters
    ----------
    obs_sets : dict of `astropy.table.Table`
        A dictionary of "observation sets". Each observation set is a table
        of observations. See the notes section below for information on what
        the table must contain.
    model : `sncosmo.Model`
        The model to use in the simulation.
    gen_params : callable
        A callable that accepts a single float (redshift) and returns
        a dictionary on each call. Typically the callable would randomly
        select parameters from some underlying distribution on each call.
    vrate : callable
        A callable that returns the SN rate per comoving volume as a
        function of redshift, in units yr^-1 Mpc^-3.
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology used to determine volume.
        The default is a FlatLambdaCDM cosmology with ``Om0=0.3``, ``H0=70.``.
    z_range : (float, float), optional
        Redshift range in which to generate transients.
    default_area : float, optional
        Area in deg^2 for observation sets that do not have an 'AREA'
        keyword in their metadata.
    nsim : int, optional
        Number of transients to simulate. Cannot set both `nsim` and `nret`.
        Default is `None`.
    nret : int, optional
        Number of transients to return (number simulated that pass
        flux significance threshold). Cannot set both `nsim` and `nret`.
        Default is 10. Set both `nsim` and `nret` to `None` to let the function
        automatically determine the number of SNe based on the area of each
        observation set and the volumetric rate.
    thresh : float, optional
        Minimum flux significant threshold for a transient to be returned.

    Returns
    -------
    sne : list of `~astropy.table.Table`
        List of tables where each table is the photometric
        data for a single simulated SN.

    Notes
    -----

    Each ``obs_set`` (values in ``obs_sets``) must have the following columns:

    * ``MJD``
    * ``FLT``
    * ``CCD_GAIN``
    * ``SKYSIG``
    * ``PSF1``
    * ``ZPTAVG``

    These are currently just what the SIMLIB files from SNANA have. In the
    future these can be more flexible.

    Examples
    --------

    Define a set of just three observations:

    >>> from astropy.table import Table
    >>> obs_set = Table({'MJD': [56176.19, 56188.254, 56207.172],
    ...                  'FLT': ['desg', 'desr', 'desi'],
    ...                  'CCD_GAIN': [1., 1., 1.],
    ...                  'SKYSIG': [91.27, 47.62, 60.40],
    ...                  'PSF1': [2.27, 2.5, 1.45],
    ...                  'ZPTAVG': [32.97, 33.05, 32.49]})
    >>> print obs_set
       MJD    ZPTAVG FLT  PSF1 SKYSIG CCD_GAIN
    --------- ------ ---- ---- ------ --------
     56176.19  32.97 desg 2.27  91.27      1.0
    56188.254  33.05 desr  2.5  47.62      1.0
    56207.172  32.49 desi 1.45   60.4      1.0
    >>> obs_sets = {0: obs_set}

    Get a model and a cosmology

    >>> import sncosmo                                    # doctest: +SKIP
    >>> from astropy import cosmology
    >>> model = sncosmo.Model(source='salt2-extended')    # doctest: +SKIP
    >>> cosmo = cosmology.FlatLambdaCDM(Om0=0.3, H0=70.)

    Get x0 corresponding to apparent mag = -19.1:

    >>> model.source.set_peakmag(-19.1, 'bessellb', 'vega')  # doctest: +SKIP
    >>> x0_0 = model.get('x0')                               # doctest: +SKIP

    Define a function that generates parameters of the model given a redshift:

    >>> from numpy.random import normal
    >>> def gen_params(z):
    ...     x1 = normal(0., 1.)
    ...     c = normal(0., 0.1)
    ...     resid = normal(0., 0.15)
    ...     hubble_offset = -0.13*x1 + 2.5*c + resid
    ...     dm = cosmo.distmod(z).value
    ...     x0 = x0_0 * 10**(-(dm + hubble_offset) / 2.5)
    ...     return {'c': c, 'x1': x1, 'x0': x0}

    Define a volumetric SN rate in SN / yr / Mpc^3:

    >>> def snrate(z):
    ...     return 0.25e-4 * (1. + 2.5 * z)

    Generate simulated SNe:

    >>> sne = simulate_vol(obs_sets, model, gen_params, snrate, cosmo=cosmo,
    ...                    nret=10)  # doctest: +SKIP
    >>> print len(sne)  # doctest: +SKIP
    10
    >>> print sne[0]  # doctest: +SKIP
       date   band      flux        fluxerr      zp  zpsys
    --------- ---- ------------- ------------- ----- -----
     56176.19 desg 780.472570859 2603.54291491 32.97    ab
    56188.254 desr 17206.2994496 1501.37068134 33.05    ab
    56207.172 desi 10323.4485412 1105.34529777 32.49    ab
    >>> print sne[0].meta  # doctest: +SKIP
    {'z': 0.52007602908199813, 'c': -0.09298497453338518,
     'x1': 1.1684716363315284, 'x0': 1.4010952818384196e-05,
     't0': 56200.279703804845}

    """

    if nsim is not None and nret is not None:
        raise ValueError('cannot specify both nsim and nret')

    # Get comoving volume in each redshift shell.
    z_bins = 100  # Good enough for now.
    z_min, z_max = z_range
    z_binedges = np.linspace(z_min, z_max, z_bins + 1)
    z_binctrs = 0.5 * (z_binedges[1:] + z_binedges[:-1])
    sphere_vols = cosmo.comoving_volume(z_binedges)
    shell_vols = sphere_vols[1:] - sphere_vols[:-1]

    # SN / (observer year) in shell
    shell_snrate = shell_vols * vrate(z_binctrs) / (1. + z_binctrs)

    # SN / (observer year) within z_binedges
    vol_snrate = np.zeros_like(z_binedges)
    vol_snrate[1:] = np.add.accumulate(shell_snrate)

    # Create a ppf (inverse cdf). We'll use this later to get
    # a random SN redshift from the distribution.
    snrate_cdf = vol_snrate / vol_snrate[-1]
    snrate_ppf = Spline1d(snrate_cdf, z_binedges, k=1)

    # Get obs sets' data, time ranges, areas and weights.
    # We do this now so we can weight the location of sne
    # according to the area and time ranges of the observation sets.
    obs_sets = obs_sets.values()
    obs_sets_data = [np.asarray(obs_set) for obs_set in obs_sets]
    time_ranges = [(obs['MJD'].min() - 10. * (1. + z_max),
                    obs['MJD'].max() + 10. * (1. + z_max))
                   for obs in obs_sets_data]
    areas = [obs_set.meta['AREA'] if 'AREA' in obs_set.meta else default_area
             for obs_set in obs_sets]
    area_time_products = [a * (t[1] - t[0])
                          for a, t in zip(areas, time_ranges)]
    total_area_time = sum(area_time_products)
    weights = [a_t / total_area_time for a_t in area_time_products]
    cumweights = np.add.accumulate(np.array(weights))

    # How many to simulate?
    if nsim is not None:
        nret = 0
    elif nret is not None:
        nsim = 0
    else:
        nsim = total_area_time / WHOLESKY_SQDEG * vol_snrate[-1]

    i = 0
    sne = []
    while i < nsim or len(sne) < nret:
        i += 1

        # which obs_set did this occur in?
        j = 0
        x = np.random.rand()
        while cumweights[j] < x:
            j += 1

        obsdata = obs_sets_data[j]
        time_range = time_ranges[j]

        # Get a redshift from the distribution
        z = snrate_ppf(np.random.rand())
        t0 = np.random.uniform(time_range[0], time_range[1])

        # Get rest of parameters from user-defined gen_params():
        params = gen_params(z)
        params.update(z=z, t0=t0)
        model.set(**params)

        # Get model fluxes
        flux = model.bandflux(obsdata['FLT'], obsdata['MJD'],
                              zp=obsdata['ZPTAVG'], zpsys='ab')

        # Get flux errors
        noise_area = 4. * math.pi * obsdata['PSF1']
        bkgpixnoise = obsdata['SKYSIG']
        fluxerr = np.sqrt((noise_area * bkgpixnoise) ** 2 +
                          np.abs(flux) / obsdata['CCD_GAIN'])

        # Scatter fluxes by the fluxerr
        flux = np.random.normal(flux, fluxerr)

        # Check if any of the fluxes are significant
        if not np.any((flux / fluxerr) > thresh):
            continue

        simulated_data = odict([('date', obsdata['MJD']),
                                ('band', obsdata['FLT']),
                                ('flux', flux),
                                ('fluxerr', fluxerr),
                                ('zp', obsdata['ZPTAVG']),
                                ('zpsys', ['ab'] * len(flux))])
        sne.append(Table(simulated_data, meta=params))

    return sne
