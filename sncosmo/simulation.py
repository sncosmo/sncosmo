"""Tools for simulation of transients."""

import copy
from collections import OrderedDict

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from numpy import random
from scipy.interpolate import InterpolatedUnivariateSpline as Spline1d

from .utils import alias_map

__all__ = ['zdist', 'realize_lcs']

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

    >>> for z in zdist(0.0, 0.25):
    ...     print(z)
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
                             for i in range(z_bins)])

    # SN / (observer year) within z_binedges
    vol_snrate = np.zeros_like(z_binedges)
    vol_snrate[1:] = np.add.accumulate(shell_snrate)

    # Create a ppf (inverse cdf). We'll use this later to get
    # a random SN redshift from the distribution.
    snrate_cdf = vol_snrate / vol_snrate[-1]
    snrate_ppf = Spline1d(snrate_cdf, z_binedges, k=1)

    # Total numbe of SNe to simulate.
    nsim = vol_snrate[-1] * (time/365.25) * (area/WHOLESKY_SQDEG)

    for i in range(random.poisson(nsim)):
        yield float(snrate_ppf(random.random()))


OBSERVATIONS_ALIASES = OrderedDict([
    ('time', set(['time', 'date', 'jd', 'mjd', 'mjdobs', 'mjd_obs'])),
    ('band', set(['band', 'bandpass', 'filter', 'flt'])),
    ('zp', set(['zp', 'zpt', 'zeropoint', 'zero_point'])),
    ('zpsys', set(['zpsys', 'zpmagsys', 'magsys'])),
    ('gain', set(['gain'])),
    ('skynoise', set(['skynoise']))
])

OBSERVATIONS_REQUIRED_ALIASES = ('time', 'band', 'zp', 'zpsys', 'gain',
                                 'skynoise')


def realize_lcs(observations, model, params, thresh=None,
                trim_observations=False, scatter=True):
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
    scatter : bool, optional
        If True, the ``flux`` value of the realized data is calculated by
        adding  a random number drawn from a Normal Distribution with a
        standard deviation equal to the ``fluxerror`` of the observation to
        the bandflux value of the observation calculated from model. Default
        is True.

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

    RESULT_COLNAMES = ('time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys')
    lcs = []

    # Copy model so we don't mess up the user's model.
    model = copy.copy(model)

    # get observations as a Table
    if not isinstance(observations, Table):
        if isinstance(observations, np.ndarray):
            observations = Table(observations)
        else:
            raise ValueError("observations not understood")

    # map column name aliases
    colname = alias_map(observations.colnames, OBSERVATIONS_ALIASES,
                        required=OBSERVATIONS_REQUIRED_ALIASES)

    # result dtype used when there are no observations
    band_dtype = observations[colname['band']].dtype
    zpsys_dtype = observations[colname['zpsys']].dtype
    result_dtype = ('f8', band_dtype, 'f8', 'f8', 'f8', zpsys_dtype)

    for p in params:
        model.set(**p)

        # Select times for output that fall within tmin amd tmax of the model
        if trim_observations:
            mask = ((observations[colname['time']] > model.mintime()) &
                    (observations[colname['time']] < model.maxtime()))
            snobs = observations[mask]
        else:
            snobs = observations

        # explicitly detect no observations and add an empty table
        if len(snobs) == 0:
            if thresh is None:
                lcs.append(Table(names=RESULT_COLNAMES,
                                 dtype=result_dtype, meta=p))
            continue

        flux = model.bandflux(snobs[colname['band']],
                              snobs[colname['time']],
                              zp=snobs[colname['zp']],
                              zpsys=snobs[colname['zpsys']])

        fluxerr = np.sqrt(snobs[colname['skynoise']]**2 +
                          np.abs(flux) / snobs[colname['gain']])

        # Scatter fluxes by the fluxerr
        # np.atleast_1d is necessary here because of an apparent bug in
        # np.random.normal: when the inputs are both length 1 arrays,
        # the output is a Python float!
        if scatter:
            flux = np.atleast_1d(np.random.normal(flux, fluxerr))

        # Check if any of the fluxes are significant
        if thresh is not None and not np.any(flux/fluxerr > thresh):
            continue

        data = [snobs[colname['time']], snobs[colname['band']], flux, fluxerr,
                snobs[colname['zp']], snobs[colname['zpsys']]]

        lcs.append(Table(data, names=RESULT_COLNAMES, meta=p))

    return lcs
