# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division

from warnings import warn

import numpy as np
from astropy.utils import OrderedDict

from .spectral import get_magsystem
from .models import get_model
from .photometric_data import PhotData

__all__ = ['fit_model']

class Result(dict):
    """Represents the optimization result.

    Notes
    -----
    This is a cut and paste from scipy, normally imported with `from
    scipy.optimize import Result`. However, it isn't available in
    scipy 0.9 (or possibly 0.10). Since this class is essentially a
    subclass of dict with attribute accessors, one can see which
    attributes are available using the `keys()` method.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"


def _guess_parvals(data, model, parnames=['t0', 'fscale']):
    """Guess parameter values based on the data, return dict"""

    nflux, nfluxerr = data.normalized_flux(zp=25., zpsys='ab',
                                           include_err=True)

    bandt0 = []
    bandfluxscale = []
    for band in np.unique(data.band):
        idx = data.band == band
        flux = nflux[idx]
        time = data.time[idx]
        weights = flux ** 2 / nfluxerr[idx]
        topn = min(len(weights) // 2, 3)
        if topn == 0: continue
        topnidx = np.argsort(weights)[-topn:]
        bandt0.append(np.average(time[topnidx], weights=weights[topnidx]))
        maxdataflux = np.average(flux[topnidx], weights=weights[topnidx])
        maxmodelflux = max(model.bandflux(band, zp=25., zpsys='ab'))
        bandfluxscale.append(maxdataflux / maxmodelflux) 

    t0 = sum(bandt0) / len(bandt0)
    fscale = sum(bandfluxscale) / len(bandfluxscale) * model.params['fscale']

    result = {}
    if 't0' in parnames: result['t0'] = t0
    if 'fscale' in parnames: result['fscale'] = fscale
    return result
                       
def fit_model(model, data, parnames, bounds=None, params_start=None,
              t0range=20., include_model_error=False, method='iminuit',
              return_minuit=False, print_level=1):
    """Fit model parameters to data by minimizing chi^2.

    Ths function defines a chi^2 to minimize, makes initial guesses for
    some standard model parameters, such as 't0' and 'fscale', based on
    the data, then runs a minimizer.

    Parameters
    ----------
    model : `~sncosmo.Model`
        The model to fit.
    data : `~numpy.ndarray` or `dict` or `~astropy.table.Table`
        Table of photometric data. Must include certain column names.
    parnames : list
        Model parameters to vary in the fit.
    bounds : `dict`, optional
        Bounded range for each parameter. Keys should be parameter names,
        values are tuples. If a bound is not given for some parameter,
        the parameter is unbounded. The exception is ``t0``, which has a
        default bound of  ``(initial guess) +/- t0range``.
    params_start : `dict`, optional
        If given, use these initial parameters in fit. Default is to use
        current model parameters.
    t0range : float, optional
        Bounds for t0 (if varied in fit and not given in `bounds`).
        Default is 20.
    method : {'iminuit', 'l-bfgs-b'}, optional
        Minimization method to use.
    return_minuit : bool, optional
        If True, and if method is 'iminuit', return the Minuit object after
        the fit.
    print_level : int, optional
        Print level. 0 is no output, 1 is standard amount.

    Returns
    -------
    res : Result
        The optimization result represented as a ``Result`` object.
        Important attributes:

        - ``params``: dictionary of best-fit parameter values.
        - ``fval``: Minimum chi squared value.
        - ``ncalls``: Number of function calls.

        See ``res.keys()`` for available attributes.

    m : `~iminuit.Minuit`
        Only returned if method is 'iminuit' and `return_minuit` is True.

    Notes
    -----
    No notes at this time.
    """
    method = method.lower()

    # Initialize data
    data = PhotData(data)

    # Get a shallow copy of the model so that we can change the parameters
    # without worrying.
    model = get_model(model, copy=True)

    # Check that 'z' is bounded (if it is going to be fit).
    if 'z' in parnames and (bounds is None or 'z' not in bounds):
        raise ValueError('z must be bounded if fit.')

    # Check redshift range to see which bands we can use in the fit.
    if 'z' not in parnames:
        valid = model.bandoverlap(data.band)
    else:
        valid = model.bandoverlap(data.band, z=bounds['z'])
        valid = np.all(valid, axis=1)
    if not np.all(valid):
        drop_bands = [repr(b) for b in set(data.band[np.invert(valid)])]
        warn("Dropping following bands from data: " + ", ".join(drop_bands) +
             "(out of model wavelength range)", RuntimeWarning)
        data = PhotData({'time': data.time[valid],
                         'band': data.band[valid],
                         'flux': data.flux[valid],
                         'fluxerr': data.fluxerr[valid],
                         'zp': data.zp[valid],
                         'zpsys': data.zpsys[valid]})

    # If we're fitting redshift and it is bounded, set initial value.
    if 'z' in parnames:
        model.set(z=(sum(bounds['z']) / 2.))

    # Get initial guesses.
    parvals0 = []
    guesses = _guess_parvals(data, model, parnames=['t0', 'fscale'])
    current = model.params
    for name in parnames:
        if params_start is not None and name in params_start:
            parvals0.append(params_start[name])
        elif name in guesses:
            parvals0.append(guesses[name])
        else:
            parvals0.append(current[name])

    # Set up a complete list of bounds.
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

    if print_level > 0:
        print "starting point:"
        for name, val, bound in zip(parnames, parvals0, bounds_list):
            print "   ", name, val, bound

    fscale_factor = 1.

    # define chi2 where input is array_like
    def chi2_array_like(parvals):
        params = dict(zip(parnames, parvals))

        if 'fscale' in params:
            params['fscale'] *= fscale_factor
        model.set(**params)

        if include_model_error:
            modelflux, modelfluxerr = model.bandflux(
                data.band, data.time, zp=data.zp, zpsys=data.zpsys,
                include_error=True)
            return np.sum((data.flux - modelflux) ** 2 /
                          (modelfluxerr ** 2 + data.fluxerr ** 2))

        else:
            modelflux = model.bandflux(
                data.band, data.time, zp=data.zp, zpsys=data.zpsys)
            return np.sum(((data.flux - modelflux) / data.fluxerr) ** 2)

    if method == 'iminuit':
        try:
            import iminuit
        except ImportError:
            raise ValueError("Minimization method 'iminuit' requires the "
                             "iminuit package")

        # The iminuit minimizer expects the function signature to have an
        # argument for each parameter.
        def chi2(*parvals):
            return chi2_array_like(parvals)

        # Set up keyword arguments to pass to Minuit initializer
        kwargs = {}
        for parname, parval, bounds in zip(parnames, parvals0, bounds_list):
            kwargs[parname] = parval
            if bounds is not None and None not in bounds:
                kwargs['limit_' + parname] = bounds
            if parname == 't0':
                step_size = 1.
            if parname == 'fscale':
                step_size = 0.1 * parval
            if parname == 'z':
                step_size = 0.05
            else:
                step_size = 1.
            kwargs['error_' + parname] = step_size

        m = iminuit.Minuit(chi2, errordef=1., forced_parameters=parnames,
                           **kwargs)
        d, l = m.migrad()
        res = Result(ncalls=d.nfcn, fval=d.fval, params=m.values,
                     errors=m.errors, covariance=m.covariance,
                     matrix=m.matrix())
        if return_minuit:
            return res, m
        else:
            return res

    elif method == 'l-bfgs-b':
        from scipy.optimize import fmin_l_bfgs_b

        # Scale 'fscale' to ~1 for numerical precision reasons.
        if 'fscale' in parnames:
            i = parnames.index('fscale')
            fscale_factor = parvals0[i]
            parvals0[i] = 1.
            if 'fscale' in bounds:
                bounds_list[i] = (bounds_list[i][0] / fscale_factor,
                                  bounds_list[i][1] / fscale_factor)

        x, f, d = fmin_l_bfgs_b(chi2_array_like, parvals0,
                                bounds=bounds_list, approx_grad=True,
                                iprint=(print_level - 1))

        d['ncalls'] = d.pop('funcalls')
        res = Result(d)
        res.params = dict(zip(parnames, x))
        res.fval = f

        # adjust fscale
        if 'fscale' in res.values:
            res.values['fscale'] *= fscale_factor

        return res

    else:
        raise ValueError('Unknown solver %s' % method)


# TODO: better name?
def mcmc_model(model, data, parnames, p0=None, nwalkers=10, nburn=200,
               nsamples=500):
    """Run an MCMC chain to get parameter contours, given the model.

    Parameters
    ----------


    Returns
    -------
    samples : `~numpy.ndarray`, shape=(nsamples*nwalkers, npar)
    """

    try:
        import emcee
    except:
        raise ImportError("mcmc_model() requires the emcee package.")


    data = PhotData(data)
    model = get_model(model)
    ndim = len(parnames)

    # TODO: Make guesses or start from p0
    
    # TODO: Define initial positions of the walkers
    p0 = np.array([55098.4 + 5. * (np.random.rand(nwalkers) - 0.5),
                   0.5 + 0.1 * (np.random.rand(nwalkers) - 0.5),
                   0. + 0.1 * (np.random.rand(nwalkers) - 0.5),
                   0. + 0.1 * (np.random.rand(nwalkers) - 0.5),
                   1.15e-17 + 0.1e-17 * (np.random.rand(nwalkers) - 0.5)]).T

    # define likelihood
    def loglikelihood(parvals):
        params = dict(zip(parnames, parvals))
        model.set(**params)
        
        modelflux = model.bandflux(
            data.band, data.time, zp=data.zp, zpsys=data.zpsys)

        return -0.5 * np.sum(((data.flux - modelflux) / data.fluxerr) ** 2)

    # Create sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglikelihood)

    # burn-in
    pos, prob, state = sampler.run_mcmc(p0, 200)
    sampler.reset()

    # production run
    sampler.run_mcmc(pos, 500)
    print "avg acceptance frac:", np.mean(sampler.acceptance_fraction)

    return sampler.flatchain
