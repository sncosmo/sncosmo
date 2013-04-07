# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division

import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from astropy.utils import OrderedDict

from spectral import get_magsystem

__all__ = ['fit_model']

def normalized_flux(data, zp=25., magsys='ab'):
    """Return flux values normalized to a common zeropoint and magnitude
    system."""

    magsys = get_magsystem(magsys)
    flux = np.empty(len(data['flux']), dtype=np.float)
    fluxerr = np.empty(len(data['flux']), dtype=np.float)

    for i in range(len(data)):
        ms = get_magsystem(data['zpsys'][i])
        factor = (ms.zpbandflux(data['band'][i]) /
                  magsys.zpbandflux(data['band'][i]) *
                  10.**(0.4 * (zp - data['zp'][i])))
        flux[i] = data['flux'][i] * factor
        fluxerr[i] = data['fluxerr'][i] * factor

    return flux, fluxerr

def guess_parvals(data, model, parnames=['t0', 'fscale']):

    nflux, nfluxerr = normalized_flux(data, zp=25., magsys='ab')

    bandt0 = []
    bandfluxscale = []
    for band in np.unique(data['band']):
        idx = data['band'] == band
        flux = nflux[idx]
        time = data['time'][idx]
        weights = flux ** 2 / nfluxerr[idx]
        topn = min(len(weights) // 2, 3)
        if topn == 0: continue
        topnidx = np.argsort(weights)[-topn:]
        bandt0.append(np.average(time[topnidx], weights=weights[topnidx]))
        maxdataflux = np.average(flux[topnidx], weights=weights[topnidx])
        maxmodelflux = max(model.bandflux(band, zp=25., zpmagsys='ab'))
        bandfluxscale.append(maxdataflux / maxmodelflux) 

    t0 = sum(bandt0) / len(bandt0)
    fscale = (sum(bandfluxscale) / len(bandfluxscale) *
                  model.params['fscale'])

    result = {}
    if 't0' in parnames: result['t0'] = t0
    if 'fscale' in parnames: result['fscale'] = fscale
    return result
                       
def fit_model(model, data, parnames, bounds=None, parvals0=None, t0range=20.,
              verbose=False, include_model_error=False):
    """Fit model parameters to data by minimizing chi^2.

    .. warn:: This function is experimental 

    Parameters
    ----------
    model : `~sncosmo.Model`
        The model to fit.
    data : `~numpy.ndarray` or `dict`
        Table containing columns 'date', 'band', 'flux', 'fluxerr', 'zp',
        'zpsys'.
    parnames : list
        Model parameters to vary in the fit.
    bounds : `dict`, optional
        Bounded range for each parameter. Keys should be parameter names,
        values are tuples. If a bound is not given for some parameter,
        the parameter is unbounded. The exception is ``t0``, which has a
        default bound of  ``(initial guess) +/- t0range``.
    parvals0 : `dict`, optional
        If given, use these initial parameters in fit. Default is to use
        current model parameters.
    verbose : bool, optional
        Print minimization info to the screen.

    Returns
    -------
    min_chisq : float
        Value of Chi^2 for fitted model parameters. The model's parameters
        are set to the best-fit parameters.

    Notes
    -----
    Uses scipy's L-BFGS-B bounded minimization algorithm.
    """

    parvals0 = []
    guesses = guess_parvals(data, model, parnames=['t0', 'fscale'])
    current = model.params
    for name in parnames:
        if name in guesses:
            parvals0.append(guesses[name])
        else:
            parvals0.append(current[name])

    if bounds is None:
        bounds = {}
    bounds_list = []
    for name in parnames:
        if name in bounds:
            bounds_list.append(bounds[name])
        elif name == 't0':
            bounds_list.append((guesses['t0'] - t0range,
                                guesses['t0'] + t0range))
        else:
            bounds_list.append((None, None))

    # If we're fitting redshift and it is bounded, set initial value
    if 'z' in parnames and 'z' in bounds:
        i = parnames.index('z')
        parvals0[i] = sum(bounds['z']) / 2.

    if verbose:
        print "starting point:"
        for name, val, bound in zip(parnames, parvals0, bounds_list):
            print "   ", name, val, bound

    # scale fscale, for numerical precision reasons.
    fscale_factor = 1.
    if 'fscale' in parnames:
        i = parnames.index('fscale')
        fscale_factor = parvals0[i]
        parvals0[i] = 1.
        if 'fscale' in bounds:
            bounds_list[i] = (bounds_list[i][0] / fscale_factor,
                              bounds_list[i][1] / fscale_factor)

    # Check redshift range to see which bands we can use in the fit.
    bands = np.unique(data['band'])
    if 'z' not in parnames:
        valid = model.bandoverlap(data['band'])
    else:
        valid = model.bandoverlap(data['band'], z=bounds['z'])
        valid = np.all(valid, axis=1)
    if not np.all(valid):
        print "WARNING: dropping following bands from data:"
        print np.unique(data['band'][np.invert(valid)])
        data = data[valid]

    def chi2(parvals):
        params = dict(zip(parnames, parvals))
        if 'fscale' in params:
            params['fscale'] *= fscale_factor
        model.set(**params)
        if include_model_error:
            modelflux, modelfluxerr = model.bandflux(
                data['band'], data['time'], zp=data['zp'],
                zpmagsys=data['zpsys'], include_error=True)
            return np.sum((data['flux'] - modelflux) ** 2 /
                          (modelfluxerr ** 2 + data['fluxerr'] ** 2))
        else:
            modelflux = model.bandflux(data['band'], data['time'],
                                       zp=data['zp'], zpmagsys=data['zpsys'])
            return np.sum(((data['flux'] - modelflux) / data['fluxerr']) ** 2)

    parvals, fval, d = fmin_l_bfgs_b(chi2, parvals0, bounds=bounds_list,
                                     approx_grad=True, iprint=(verbose - 1))
    params = dict(zip(parnames, parvals))
    if 'fscale' in params:
        params['fscale'] *= fscale_factor
    model.set(**params)
    return fval
