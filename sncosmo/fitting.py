# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division
from warnings import warn

import numpy as np

from .spectral import get_magsystem, get_bandpass
from .models import get_model
from .photometric_data import standardize_data, normalize_data
from . import nest

__all__ = ['fit_lc', 'sample_lc', 'mcmc_lc']

class Result(dict):
    """Represents the optimization result.

    Notes
    -----
    This is a cut and paste from scipy, normally imported with `from
    scipy.optimize import Result`. However, it isn't available in
    scipy 0.9 (or possibly 0.10), so it is included here.
    Since this class is essentially a subclass of dict with attribute
    accessors, one can see which attributes are available using the
    `keys()` method.
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


def guess_parvals(data, model, parnames=['t0', 'fscale']):
    """Guess parameter values based on the data, return dict.
    The maximum of `model.bandflux` and the maximum of the data
    are used to guess `fscale`. The current settings of the parameters
    't0' and 'fscale' do not affect the results. Other parameters, such as 
    'z' might affect the guesses somewhat.

    Assume that data has already been normalized.
    """

    band_tmax = []
    band_fscale = []
    for band in set(data['band'].tolist()):
        idx = data['band'] == band
        time = data['time'][idx]
        flux = data['flux'][idx]
        fluxerr = data['fluxerr'][idx]

        weights = flux * np.abs(flux / fluxerr)
        topn = min(len(weights) // 2, 3)
        if topn == 0:
            continue
        topnidx = np.argsort(weights)[-topn:]
        band_tmax.append(np.average(time[topnidx],
                                    weights=weights[topnidx]))
        maxdataflux = np.average(flux[topnidx], weights=weights[topnidx])
        maxmodelflux = np.max(model.bandflux(band, zp=data['zp'][0],
                                             zpsys=data['zpsys'][0]))
        band_fscale.append(maxdataflux / maxmodelflux *
                           model._params['fscale']) 

    result = {}
    if 't0' in parnames:
        result['t0'] = sum(band_tmax) / len(band_tmax) - model.refphase
    if 'fscale' in parnames:
        result['fscale'] = sum(band_fscale) / len(band_fscale)

    # check that guessing succeeded
    if any([np.isnan(v) or np.isinf(v) for v in result.values()]):
        raise RuntimeError('Parameter guessing failed. Check data values.')

    return result

def fit_lc(data, model, parnames, p0=None, bounds=None,
           t0_range=20., include_model_error=False,
           fit_offset=False, offset_zp=25., offset_zpsys='ab',
           method='iminuit', return_minuit=False, max_ncall=10000,
           print_level=0):
    """Fit model parameters to data by minimizing chi^2.

    Ths function defines a chi^2 to minimize, makes initial guesses for
    some standard model parameters, such as 't0' and 'fscale', based on
    the data, then runs a minimizer.

    Parameters
    ----------
    data : `~numpy.ndarray` or `dict` of list_like
        Table of photometric data. Must include certain column names.
    model : `~sncosmo.Model`
        The model to fit.
    parnames : list
        Model parameters to vary in the fit.
    p0 : `dict`, optional
        If given, use these initial parameters in fit. Default is to use
        current model parameters.
    bounds : `dict`, optional
        Bounded range for each parameter. Keys should be parameter names,
        values are tuples. If a bound is not given for some parameter,
        the parameter is unbounded. The exception is ``t0``, which has a
        default bound of  ``(initial guess) +/- t0_range``.
    t0_range : float, optional
        Bounds for t0 (if varied in fit and not given in `bounds`).
        Default is 20.
    include_model_error : bool, optional
        Default is False.
    fit_offset : bool or list of str, optional
        Fit for the "offset flux value" in each bandpass
        specified. The "offset flux value" is added to the model flux
        in the fit. If a list is supplied, it should be a list of bandpass
        names for which to fit the offset. If a bool is supplied, then
        the offset will be fit for all bandpasses if True or for no
        bandpasses if False. The default is False.
    offset_zp : float
        Default is 25.
    offset_zpsys : `~sncosmo.MagSystem` or str, optional
        Default is 'ab'.
    method : {'iminuit', 'l-bfgs-b'}, optional
        Minimization method to use.
    return_minuit : bool, optional
        For method 'iminuit', return the `~iminuit.Minuit` object used in
        the fit.
    print_level : int, optional
        Print level. 0 is no output, 1 is standard amount.

    Returns
    -------
    res : Result
        The optimization result represented as a ``Result`` object (a dict
        subclass with attribute access). Some important attributes:

        - ``res.params``: dictionary of best-fit parameter values.
        - ``res.fval``: Minimum chi squared value.
        - ``res.ncalls``: Number of function calls.

        See ``res.keys()`` for available attributes.

    m : `~iminuit.Minuit`
        Only returned if method is 'iminuit' and `return_minuit` is True.

    Notes
    -----
    No notes at this time.
    """

    data = standardize_data(data)
    unique_bandnames = [get_bandpass(band).name
                        for band in set(data['band'].tolist())]

    method = method.lower()
    if fit_offset == True:
        fit_offset = unique_bandnames

    if fit_offset:
        data = normalize_data(data, zp=offset_zp, zpsys=offset_zpsys)

    # Get a shallow copy of the model so that we can change the parameters
    # without worrying.
    model = get_model(model, copy=True)

    # Check that 'z' is bounded (if it is going to be fit).
    if 'z' in parnames and (bounds is None or 'z' not in bounds):
        raise ValueError('z must be bounded if fit.')

    # Cut bands that are not allowed by the wavelength range of the model
    if 'z' not in parnames:
        valid = model.bandoverlap(data['band'])
    else:
        valid = model.bandoverlap(data['band'], z=bounds['z'])
        valid = np.all(valid, axis=1)
    if not np.all(valid):
        drop_bands = [repr(b) for b in set(data['band'][np.invert(valid)])]
        warn("Dropping following bands from data: " + ", ".join(drop_bands) +
             "(out of model wavelength range)", RuntimeWarning)
        data = data[valid]

    # Set initial parameters, to help with guessing parameters, below.
    if p0 is None:
        p0 = {}
    if 'z' in parnames and 'z' not in p0:
        p0['z'] = sum(bounds['z']) / 2.
    model.set(**p0)

    # Get list of initial guesses.
    if fit_offset:
        guesses = guess_parvals(data, model, parnames=['t0', 'fscale'])
    else:
        ndata = normalize_data(data, zp=offset_zp, zpsys=offset_zpsys)
        guesses = guess_parvals(ndata, model, parnames=['t0', 'fscale'])

    # Set initial parameters. Order of priority: 
    #   1. p0
    #   2. guesses
    #   3. current params (if not None)
    #   4. 0.
    parvals0 = []
    current = model.params
    for name in parnames:
        if name in p0:
            parvals0.append(p0[name])
        elif name in guesses:
            parvals0.append(guesses[name])
        elif current[name] is not None:
            parvals0.append(current[name])
        else:
            parvals0.append(0.)

    # Add parameters for offset, if we're fitting it.
    # TODO: Make this work with Bandpass objects not in registry.
    if fit_offset:
        offset_to_data = {} # map offset param to data idx
        for bandname in fit_offset:
            parname = 'offset_' + bandname
            idx = data['band'] == bandname
            parnames.append(parname)
            parvals0.append(np.min(data['flux'][idx]))
            offset_to_data[parname] = idx

    # Set up a complete list of bounds.
    bounds_list = []
    for name in parnames:
        if bounds is not None and name in bounds:
            bounds_list.append(bounds[name])
        elif name == 't0':
            i = parnames.index('t0')
            bounds_list.append((parvals0[i] - t0_range,
                                parvals0[i] + t0_range))
        else:
            bounds_list.append((None, None))

    # count degrees of freedom
    ndof = len(data) - len(parnames)

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
            mflux, mfluxerr = \
                model.bandflux(data['band'], data['time'], zp=data['zp'],
                               zpsys=data['zpsys'], include_error=True)
            denom = mfluxerr**2 + data['fluxerr']**2

        else:
            mflux = model.bandflux(data['band'], data['time'],
                                   zp=data['zp'], zpsys=data['zpsys'])
            denom = data['fluxerr']**2

        if fit_offset:
            for key, value in params.iteritems():
                if key[0:7] == 'offset_':
                    idx = offset_to_data[key]
                    mflux[idx] += value

        return np.sum((data['flux'] - mflux)**2 / denom)

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
                           print_level=print_level, **kwargs)
        d, l = m.migrad(ncall=max_ncall)
        res = Result(ncalls=d.nfcn, fval=d.fval, params=m.values,
                     errors=m.errors, covariance=m.covariance,
                     matrix=m.matrix(), ndof=ndof)

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
        res.ndof = ndof

        # adjust fscale
        if 'fscale' in res.values:
            res.values['fscale'] *= fscale_factor

    else:
        raise ValueError('Unknown solver %s' % method)

    # append offsets to result by bandname
    if fit_offset:
        res.offsets = {}
        for key, value in res.params.iteritems():
            if key[0:7] == 'offset_':
                res.offsets[key[7:]] = value

    if method == 'iminuit' and return_minuit:
        return res, m
    else:
        return res

def sample_lc(data, model, param_bounds, nobj=100, maxiter=10000,
              verbose=False):

    data = standardize_data(data)

    # which parameters are we varying?
    names, bounds = param_bounds.keys(), param_bounds.values()
    idx = np.array([model.param_names.index(name) for name in names])
    v0 = np.array([bound[0] for bound in bounds])
    vdiff = np.array([bound[1] - bound[0] for bound in bounds])

    def prior(u):
        return v0 + u*vdiff

    def loglikelihood(parameters):
        model.parameters[idx] = parameters
        mflux = model.bandflux(data['band'], data['time'],
                               zp=data['zp'], zpsys=data['zpsys'])
        chisq = np.sum(((data['flux'] - mflux) / data['fluxerr'])**2)
        return -chisq / 2.

    res = nest.nest(loglikelihood, prior, len(idx), nobj=nobj,
                    maxiter=maxiter, verbose=verbose)
    return Result(res)

def mcmc_lc(data, model, parnames, p0=None, errors=None, nwalkers=10,
            nburn=100, nsamples=500, return_sampler=False, verbose=False):
    """Run an MCMC chain to get model parameter samples.

    Parameters
    ----------
    data : `~numpy.ndarray` or `dict` of list_like
        Table of photometric data. Must include certain column names.
    model : `~sncosmo.Model`
        The model to fit.
    parnames : list
        Model parameters to vary in the fit.
    p0 : `dict`, optional
        If given, use these initial parameters in fit. Default is to use
        current model parameters.
    errors : `dict`, optional
    nwalkers : int, optional
    nburn : int, optional
    nsamples : int, optional
    return_sampler : bool, optional
    verbose : bool, optional

    Returns
    -------
    samples : `~numpy.ndarray`
        The shape is (nsamples * nwalkers, npar).
    """

    try:
        import emcee
    except:
        raise ImportError("mcmc_lc() requires the emcee package.")

    data = standardize_data(data)
    model = get_model(model)
    ndim = len(parnames)

    # --------------------- COPIED FROM FIT_LC ----------------------------
    # Set initial parameters, to help with guessing parameters, below.
    if p0 is not None:
        model.set(**p0)

    # Get list of initial guesses.
    ndata = normalize_data(data, zp=25., zpsys='ab')
    guesses = guess_parvals(ndata, model, parnames=['t0', 'fscale'])

    # Set initial parameters. Order of priority: 
    #   1. p0
    #   2. guesses
    #   3. current params (if not None)
    #   4. 0.
    parvals0 = []
    current = model.params
    for name in parnames:
        if p0 is not None and name in p0:
            parvals0.append(p0[name])
        elif name in guesses:
            parvals0.append(guesses[name])
        elif current[name] is not None:
            parvals0.append(current[name])
        else:
            parvals0.append(0.)
    # --------------------- END OF COPY FROM FIT_LC ------------------------

    step_sizes = []
    for parname, parval in zip(parnames, parvals0):
        if errors is not None and parname in errors:
            step_size = errors[parname]
        elif parname == 't0':
            step_size = 0.5
        elif parname == 'fscale':
            step_size = 0.1 * parval
        elif parname == 'z':
            step_size = 0.05
        else:
            step_size = 0.1
        step_sizes.append(step_size)

    # Starting positions of walkers.
    randarr = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
    start = np.array(parvals0) + np.array(step_sizes) * (randarr - 0.5)

    # define likelihood
    def loglikelihood(parvals):
        params = dict(zip(parnames, parvals))
        model.set(**params)
        mflux = model.bandflux(data['band'], data['time'],
                               zp=data['zp'], zpsys=data['zpsys'])
        return -0.5 * np.sum(((data['flux'] - mflux) / data['fluxerr']) ** 2)

    # Create sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglikelihood)

    # burn-in
    pos, prob, state = sampler.run_mcmc(start, nburn)
    sampler.reset()

    # production run
    sampler.run_mcmc(pos, nsamples)
    if verbose:
        print "Avg acceptance fraction:", np.mean(sampler.acceptance_fraction)

    if return_sampler:
        return sampler
    else:
        return sampler.flatchain

