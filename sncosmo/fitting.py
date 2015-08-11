# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division, print_function
from warnings import warn
import copy

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline1d
from astropy.utils import OrderedDict as odict
from astropy.extern import six

from .photdata import standardize_data, normalize_data
from . import nest
from .utils import Result, Interp1D, ppf, weightedcov

__all__ = ['fit_lc', 'nest_lc', 'mcmc_lc', 'flatten_result', 'chisq']


class DataQualityError(Exception):
    pass


def _chisq(data, model, modelcov):
    """Like chisq but assumes data is already standardized.

    The purpose of having this as a separate function is for the benefit
    of fitting functions that standardize data once and call chisq many
    times. Such functions explicitly call standardize_data and then this
    function.
    """

    if modelcov:
        mflux, mcov = model.bandfluxcov(data['band'], data['time'],
                                        zp=data['zp'], zpsys=data['zpsys'])
        diff = (data['flux'] - mflux)
        totcov = mcov + np.diag(data['fluxerr']**2)
        invtotcov = np.linalg.inv(totcov)
        return np.dot(np.dot(diff[np.newaxis, :], invtotcov),
                      diff[:, np.newaxis])[0, 0]
    else:
        mflux = model.bandflux(data['band'], data['time'],
                               zp=data['zp'], zpsys=data['zpsys'])
        return np.sum(((data['flux'] - mflux) / data['fluxerr'])**2)


def chisq(data, model, modelcov=False):
    """Calculate chisq statistic for the model, given the data.

    Parameters
    ----------
    model : `~sncosmo.Model`
    data : `~astropy.table.Table` or `~numpy.ndarray` or `dict`
        Table of photometric data. Must include certain columns.
        See the "Photometric Data" section of the documentation for
        required columns.
    modelcov : bool
        Include model covariance? Calls ``model.bandfluxcov`` method
        instead of ``model.bandflux``. The source in the model must therefore
        implement covariance.

    Returns
    -------
    chisq : float
    """
    data = standardize_data(data)
    return _chisq(data, model, modelcov=modelcov)


def flatten_result(res):
    """Turn a result from fit_lc into a simple dictionary of key, value pairs.

    Useful when saving results to a text file table, where structures
    like a covariance matrix cannot be easily written to a single
    table row.

    Parameters
    ----------
    res : Result
        Result object from `~sncosmo.fit_lc`.

    Returns
    -------
    flatres : Result
        Flattened result. Keys are all strings, values are one of: float, int,
        string), suitable for saving to a text file.
    """

    flat = Result(success=(1 if res.success else 0),
                  ncall=res.ncall,
                  chisq=res.chisq,
                  ndof=res.ndof)

    # Parameters and uncertainties
    for i, n in enumerate(res.param_names):
        flat[n] = res.parameters[i]
        if res.errors is None:
            flat[n + '_err'] = float('nan')
        else:
            flat[n + '_err'] = res.errors.get(n, 0.)

    # Covariances.
    for n1 in res.param_names:
        for n2 in res.param_names:
            key = n1 + '_' + n2 + '_cov'
            if n1 not in res.cov_names or n2 not in res.cov_names:
                flat[key] = 0.
            elif res.covariance is None:
                flat[key] = float('nan')
            else:
                i = res.cov_names.index(n1)
                j = res.cov_names.index(n2)
                flat[key] = res.covariance[i, j]

    return flat


def cut_bands(data, model, z_bounds=None):

    if z_bounds is None:
        valid = model.bandoverlap(data['band'])
    else:
        valid = model.bandoverlap(data['band'], z=z_bounds)
        valid = np.all(valid, axis=1)

    if not np.all(valid):
        # Fail if there are no overlapping bands whatsoever.
        if not np.any(valid):
            raise RuntimeError('No bands in data overlap the model.')

        # Otherwise, warn that we are dropping some bands from the data:
        drop_bands = [repr(b) for b in set(data['band'][np.invert(valid)])]
        warn("Dropping following bands from data: " + ", ".join(drop_bands) +
             "(out of model wavelength range)", RuntimeWarning)
        data = data[valid]

    return data


def t0_bounds(data, model):
    """Determine bounds on t0 parameter of the model.

    The lower bound is such that the latest model time is equal to the
    earliest data time. The upper bound is such that the earliest
    model time is equal to the latest data time.

    Assumes the data has been standardized."""

    return (model.get('t0') + np.min(data['time']) - model.maxtime(),
            model.get('t0') + np.max(data['time']) - model.mintime())


def guess_t0_and_amplitude(data, model, minsnr):
    """Guess t0 and amplitude of the model based on the data.

    Assumes the data has been standardized."""

    timegrid = np.linspace(model.mintime(), model.maxtime(),
                           int(model.maxtime() - model.mintime() + 1))

    snr = data['flux'] / data['fluxerr']
    significant_data = data[snr > minsnr]
    modelflux = {}
    dataflux = {}
    datatime = {}
    zp = data['zp'][0]  # Same for all entries in "standardized" data.
    zpsys = data['zpsys'][0]  # Same for all entries in "standardized" data.
    for band in set(data['band']):
        mask = significant_data['band'] == band
        if np.any(mask):
            modelflux[band] = (
                model.bandflux(band, timegrid, zp=zp, zpsys=zpsys) /
                model.parameters[2])
            dataflux[band] = significant_data['flux'][mask]
            datatime[band] = significant_data['time'][mask]

    if len(modelflux) == 0:
        raise DataQualityError('No data points with S/N > {0}. Initial '
                               'guessing failed.'.format(minsnr))

    # find band with biggest ratio of maximum data flux to maximum model flux
    maxratio = float("-inf")
    maxband = None
    for band in modelflux:
        ratio = np.max(dataflux[band]) / np.max(modelflux[band])
        if ratio > maxratio:
            maxratio = ratio
            maxband = band

    # amplitude guess is the largest ratio
    amplitude = abs(maxratio)

    # time guess is time of max in the band with the biggest ratio
    data_tmax = datatime[maxband][np.argmax(dataflux[maxband])]
    model_tmax = timegrid[np.argmax(modelflux[maxband])]
    t0 = model.get('t0') + data_tmax - model_tmax

    return t0, amplitude


def fit_lc(data, model, vparam_names, bounds=None, method='minuit',
           guess_amplitude=True, guess_t0=True, guess_z=True,
           minsnr=5., modelcov=False, verbose=False, maxcall=10000,
           **kwargs):
    """Fit model parameters to data by minimizing chi^2.

    Ths function defines a chi^2 to minimize, makes initial guesses for
    t0 and amplitude, then runs a minimizer.

    Parameters
    ----------
    data : `~astropy.table.Table` or `~numpy.ndarray` or `dict`
        Table of photometric data. Must include certain columns.
        See the "Photometric Data" section of the documentation for
        required columns.
    model : `~sncosmo.Model`
        The model to fit.
    vparam_names : list
        Model parameters to vary in the fit.
    bounds : `dict`, optional
        Bounded range for each parameter. Keys should be parameter
        names, values are tuples. If a bound is not given for some
        parameter, the parameter is unbounded. The exception is
        ``t0``: by default, the minimum bound is such that the latest
        phase of the model lines up with the earliest data point and
        the maximum bound is such that the earliest phase of the model
        lines up with the latest data point.
    guess_amplitude : bool, optional
        Whether or not to guess the amplitude from the data. If false, the
        current model amplitude is taken as the initial value. Only has an
        effect when fitting amplitude. Default is True.
    guess_t0 : bool, optional
        Whether or not to guess t0. Only has an effect when fitting t0.
        Default is True.
    guess_z : bool, optional
        Whether or not to guess z (redshift). Only has an effect when fitting
        redshift. Default is True.
    minsnr : float, optional
        When guessing amplitude and t0, only use data with signal-to-noise
        ratio (flux / fluxerr) greater than this value. Default is 5.
    method : {'minuit'}, optional
        Minimization method to use. Currently there is only one choice.
    modelcov : bool, optional
        Include model covariance when calculating chisq. Default is False.
    verbose : bool, optional
        Print messages during fitting.

    Returns
    -------
    res : Result
        The optimization result represented as a ``Result`` object, which is
        a `dict` subclass with attribute access. Therefore, ``res.keys()``
        provides a list of the attributes. Attributes are:

        - ``success``: boolean describing whether fit succeeded.
        - ``message``: string with more information about exit status.
        - ``ncall``: number of function evaluations.
        - ``chisq``: minimum chi^2 value.
        - ``ndof``: number of degrees of freedom
          (len(data) - len(vparam_names)).
        - ``param_names``: same as ``model.param_names``.
        - ``parameters``: 1-d `~numpy.ndarray` of best-fit values
          (including fixed parameters) corresponding to ``param_names``.
        - ``vparam_names``: list of varied parameter names.
        - ``covariance``: 2-d `~numpy.ndarray` of parameter covariance;
          indicies correspond to order of ``vparam_names``.
        - ``errors``: OrderedDict of varied parameter uncertainties.
          Corresponds to square root of diagonal entries in covariance matrix.

    fitmodel : `~sncosmo.Model`
        A copy of the model with parameters set to best-fit values.

    Notes
    -----

    **t0 guess:** If ``t0`` is being fit and ``guess_t0=True``, the
    function will guess the initial starting point for ``t0`` based on
    the data. The guess is made as follows:

    * Evaluate the time and value of peak flux for the model in each band
      given the current model parameters.
    * Determine the data point with maximum flux in each band, for points
      with signal-to-noise ratio > ``minsnr`` (default is 5). If no points
      meet this criteria, the band is ignored (for the purpose of guessing
      only).
    * For each band, compare model's peak flux to the peak data point. Choose
      the band with the highest ratio of data / model.
    * Set ``t0`` so that the model's time of peak in the chosen band
      corresponds to the peak data point in this band.

    **amplitude guess:** If amplitude (assumed to be the first model parameter)
    is being fit and ``guess_amplitude=True``, the function will guess the
    initial starting point for the amplitude based on the data.

    **redshift guess:** If redshift (``z``) is being fit and ``guess_z=True``,
    the function will set the initial value of ``z`` to the average of the
    bounds on ``z``.

    Examples
    --------

    The `~sncosmo.flatten_result` function can be used to make the result
    a dictionary suitable for appending as rows of a table:

    >>> from astropy.table import Table               # doctest: +SKIP
    >>> table_rows = []                               # doctest: +SKIP
    >>> for sn in sne:                                # doctest: +SKIP
    ...     res, fitmodel = sncosmo.fit_lc(           # doctest: +SKIP
    ...          sn, model, ['t0', 'x0', 'x1', 'c'])  # doctest: +SKIP
    ...     table_rows.append(flatten_result(res))    # doctest: +SKIP
    >>> t = Table(table_rows)                         # doctest: +SKIP

    """

    # Standardize and normalize data.
    data = standardize_data(data)
    data = normalize_data(data)

    # Make a copy of the model so we can modify it with impunity.
    model = copy.copy(model)

    # Check that vparam_names isn't empty and contains only parameters
    # known to the model.
    if len(vparam_names) == 0:
        raise ValueError("no parameters supplied")
    for s in vparam_names:
        if s not in model.param_names:
            raise ValueError("Parameter not in model: " + repr(s))

    # Order vparam_names the same way it is ordered in the model:
    vparam_names = [s for s in model.param_names if s in vparam_names]

    # initialize bounds
    if bounds is None:
        bounds = {}

    # Check that 'z' is bounded (if it is going to be fit).
    if 'z' in vparam_names:
        if 'z' not in bounds or None in bounds['z']:
            raise ValueError('z must be bounded if fit.')
        if guess_z:
            model.set(z=sum(bounds['z']) / 2.)
        if model.get('z') < bounds['z'][0] or model.get('z') > bounds['z'][1]:
            raise ValueError('z out of range.')

    # Cut bands that are not allowed by the wavelength range of the model.
    data = cut_bands(data, model, z_bounds=bounds.get('z', None))

    # Unique set of bands in data
    bands = set(data['band'].tolist())

    # Find t0 bounds to use, if not explicitly given
    if 't0' in vparam_names and 't0' not in bounds:
        bounds['t0'] = t0_bounds(data, model)

    # Note that in the parameter guessing below, we assume that the source
    # amplitude is the 3rd parameter of the Model (1st parameter of the Source)

    # Turn off guessing if we're not fitting the parameter.
    if model.param_names[2] not in vparam_names:
        guess_amplitude = False
    if 't0' not in vparam_names:
        guess_t0 = False

    # Make guesses for t0 and amplitude.
    # (For now, we assume it is the 3rd parameter of the model.)
    if (guess_amplitude or guess_t0):
        t0, amplitude = guess_t0_and_amplitude(data, model, minsnr)
        if guess_amplitude:
            model.parameters[2] = amplitude
        if guess_t0:
            model.set(t0=t0)

    # count degrees of freedom
    ndof = len(data) - len(vparam_names)

    if method == 'minuit':
        try:
            import iminuit
        except ImportError:
            raise ValueError("Minimization method 'minuit' requires the "
                             "iminuit package")

        # The iminuit minimizer expects the function signature to have an
        # argument for each parameter.
        def fitchisq(*parameters):
            model.parameters = parameters
            return _chisq(data, model, modelcov=modelcov)

        # Set up keyword arguments to pass to Minuit initializer.
        kwargs = {}
        for name in model.param_names:
            kwargs[name] = model.get(name)  # Starting point.

            # Fix parameters not being varied in the fit.
            if name not in vparam_names:
                kwargs['fix_' + name] = True
                kwargs['error_' + name] = 0.
                continue

            # Bounds
            if name in bounds:
                if None in bounds[name]:
                    raise ValueError('one-sided bounds not allowed for '
                                     'minuit minimizer')
                kwargs['limit_' + name] = bounds[name]

            # Initial step size
            if name in bounds:
                step = 0.02 * (bounds[name][1] - bounds[name][0])
            elif model.get(name) != 0.:
                step = 0.1 * model.get(name)
            else:
                step = 1.
            kwargs['error_' + name] = step

        if verbose:
            print("Initial parameters:")
            for name in vparam_names:
                print(name, kwargs[name], 'step=', kwargs['error_' + name],
                      end=" ")
                if 'limit_' + name in kwargs:
                    print('bounds=', kwargs['limit_' + name], end=" ")
                print()

        m = iminuit.Minuit(fitchisq, errordef=1.,
                           forced_parameters=model.param_names,
                           print_level=(1 if verbose else 0),
                           throw_nan=True, **kwargs)
        d, l = m.migrad(ncall=maxcall)

        # Build a message.
        message = []
        if d.has_reached_call_limit:
            message.append('Reached call limit.')
        if d.hesse_failed:
            message.append('Hesse Failed.')
        if not d.has_covariance:
            message.append('No covariance.')
        elif not d.has_accurate_covar:  # iminuit docs wrong
            message.append('Covariance may not be accurate.')
        if not d.has_posdef_covar:  # iminuit docs wrong
            message.append('Covariance not positive definite.')
        if d.has_made_posdef_covar:
            message.append('Covariance forced positive definite.')
        if not d.has_valid_parameters:
            message.append('Parameter(s) value and/or error invalid.')
        if len(message) == 0:
            message.append('Minimization exited successfully.')
        # iminuit: m.np_matrix() doesn't work

        # numpy array of best-fit values (including fixed parameters).
        parameters = np.array([m.values[name] for name in model.param_names])
        model.parameters = parameters  # set model parameters to best fit.

        # Covariance matrix (only varied parameters) as numpy array.
        if m.covariance is None:
            covariance = None
        else:
            covariance = np.array([
                [m.covariance[(n1, n2)] for n1 in vparam_names]
                for n2 in vparam_names])

        # OrderedDict of errors
        if m.errors is None:
            errors = None
        else:
            errors = odict([(name, m.errors[name]) for name in vparam_names])

        # Compile results
        res = Result(success=d.is_valid,
                     message=' '.join(message),
                     ncall=d.nfcn,
                     chisq=d.fval,
                     ndof=ndof,
                     param_names=model.param_names,
                     parameters=parameters,
                     vparam_names=vparam_names,
                     covariance=covariance,
                     errors=errors)

        # TODO remove cov_names in a future release.
        depmsg = ("The `cov_names` attribute is deprecated in sncosmo v1.0 "
                  "and will be removed in v1.1. Use `vparam_names` instead.")
        res.__dict__['deprecated']['cov_names'] = (vparam_names, depmsg)

    else:
        raise ValueError("unknown method {0:r}".format(method))

    # TODO remove this in a future release.
    if "flatten" in kwargs:
        warnings.warn("The `flatten` keyword is deprecated in sncosmo v1.0 "
                      "and will be removed in v1.1. Use the flatten_result() "
                      "function instead.")
        if kwargs["flatten"]:
            res = flatten_result(res)
    return res, model


# ---------------------------------------------------------------------
# This is the code for adding tied parameters to results of nest_lc
# before returning

#    # Add tied parameters to results. This is inelegant, but, eh.
#    nsamples = len(res['samples_parvals'])
#    res['nsamples'] = nsamples
#    if tied is not None:
#        tiedparnames = tied.keys()
#        ntiedpar = len(tiedparnames)
#        tiedparvals = np.empty((nsamples, ntiedpar), dtype=np.float)
#        for i in range(nsamples):
#            d = dict(zip(parnames, res['samples_parvals'][i, :]))
#            for j, parname in enumerate(tiedparnames):
#                tiedparvals[i, j] = tied[parname](d)
#
#        res['samples_parvals'] = np.hstack((res['samples_parvals'],
#                                            tiedparvals))
#        parnames = parnames + tiedparnames
#
#    # Sample averages and their standard deviations.
#    res['parvals'] = np.average(res['samples_parvals'],
#                                weights=res['samples_wt'], axis=0)
#    res['parerrs'] = np.sqrt(np.sum(res['samples_wt'][:, np.newaxis] *
#                             res['samples_parvals']**2, axis=0) -
#                             res['parvals']**2)
#
#    # Add some more to results
#    res['parnames'] = parnames
#    res['dof'] = len(data) - npar
#
#    return res


def _nest_lc(data, model, vparam_names, modelcov,
             bounds=None, priors=None, ppfs=None, tied=None,
             nobj=100, maxiter=10000, maxcall=1000000, verbose=False):
    """Assumes that data has already been standardized.
    (do `data = standardize_data(data)`).  Output samples and parameter
    names are not reordered (order wil be the same as input parameter list).
    """

    if ppfs is None:
        ppfs = {}
    if tied is None:
        tied = {}

    # Convert bounds/priors combinations into ppfs
    if bounds is not None:
        for key, val in six.iteritems(bounds):
            if key in ppfs:
                continue  # ppfs take priority over bounds/priors
            a, b = val
            if priors is not None and key in priors:
                # solve ppf at discrete points and return interpolating
                # function
                x_samples = np.linspace(0., 1., 101)
                ppf_samples = ppf(priors[key], x_samples, a, b)
                f = Interp1D(0., 1., ppf_samples)
            else:
                f = Interp1D(0., 1., np.array([a, b]))
            ppfs[key] = f

    # NOTE: It is important that iparam_names is in the same order
    # every time, otherwise results will not be reproducible, even
    # with same random seed.  This is because iparam_names[i] is
    # matched to u[i] below and u will be in a reproducible order,
    # so iparam_names must also be.
    iparam_names = [key for key in vparam_names if key in ppfs]
    ppflist = [ppfs[key] for key in iparam_names]
    nipar = len(iparam_names)  # length of u
    npar = len(vparam_names)  # length of v

    # Check that all param_names either have a direct prior or are tied.
    for name in vparam_names:
        if name in iparam_names:
            continue
        if name in tied:
            continue
        raise ValueError("Must supply ppf or bounds or tied for parameter '{}'"
                         .format(name))

    def prior(u):
        d = {}
        for i in range(nipar):
            d[iparam_names[i]] = ppflist[i](u[i])
        v = np.empty(npar, dtype=np.float)
        for i in range(npar):
            key = vparam_names[i]
            if key in d:
                v[i] = d[key]
            else:
                v[i] = tied[key](d)
        return v

    # Indicies of the model parameters in vparam_names
    idx = np.array([model.param_names.index(name) for name in vparam_names])

    def loglikelihood(parameters):
        model.parameters[idx] = parameters
        return -0.5 * _chisq(data, model, modelcov=modelcov)

    res = nest.nest(loglikelihood, prior, npar, nipar, nobj=nobj,
                    maxiter=maxiter, maxcall=maxcall, verbose=verbose)
    res.vparam_names = copy.copy(vparam_names)
    res.ndof = len(data) - len(vparam_names)
    return res


def nest_lc(data, model, vparam_names, bounds, guess_amplitude_bound=False,
            minsnr=5., priors=None, ppfs=None, nobj=100, maxiter=10000,
            maxcall=1000000, modelcov=False, verbose=False):
    """Run nested sampling algorithm to estimate model parameters and evidence.

    Parameters
    ----------
    data : `~astropy.table.Table` or `~numpy.ndarray` or `dict`
        Table of photometric data. Must include certain columns.
        See the "Photometric Data" section of the documentation for
        required columns.
    model : `~sncosmo.Model`
        The model to fit.
    vparam_names : list
        Model parameters to vary in the fit.
    bounds : `dict`
        Bounded range for each parameter. Bounds must be given for
        each parameter, with the exception of ``t0``: by default, the
        minimum bound is such that the latest phase of the model lines
        up with the earliest data point and the maximum bound is such
        that the earliest phase of the model lines up with the latest
        data point.
    guess_amplitude_bound : bool, optional
        If true, bounds for the model's amplitude parameter are determined
        automatically based on the data and do not need to be included in
        `bounds`. The lower limit is set to zero and the upper limit is 10
        times the amplitude "guess" (which is based on the highest-flux
        data point in any band). Default is False.
    minsnr : float, optional
        Minimum signal-to-noise ratio of data points to use when guessing
        amplitude bound. Default is 5.
    priors : `dict`, optional
        Prior probability distribution function for each parameter. The keys
        should be parameter names and the values should be callables that
        accept a float. If a parameter is not in the dictionary, the prior
        defaults to a flat distribution between the bounds.
    ppfs : `dict`, optional
        Prior percent point function (inverse of the cumulative distribution
        function) for each parameter. If a parameter is in this dictionary,
        the ppf takes precedence over a prior pdf specified in ``priors``.
    nobj : int, optional
        Number of objects (e.g., concurrent sample points) to use. Increasing
        nobj increases the accuracy (due to denser sampling) and also the time
        to solution.
    maxiter : int, optional
        Maximum number of iterations. Default is 10000.
    maxcall : int, optional
        Maximum number of likelihood evaluations. Default is 1000000.
    modelcov : bool, optional
        Include model covariance when calculating chisq. Default is False.
    verbose : bool, optional

    Returns
    -------
    res : Result
        Attributes are:

        * ``niter``: total number of iterations
        * ``ncall``: total number of likelihood function calls
        * ``time``: time in seconds spent in iteration loop.
        * ``logz``: natural log of the Bayesian evidence Z.
        * ``logzerr``: estimate of uncertainty in logz (due to finite sampling)
        * ``h``: Bayesian information.
        * ``vparam_names``: list of parameter names varied.
        * ``samples``: 2-d `~numpy.ndarray`, shape is (nsamples, nparameters).
          Each row is the parameter values for a single sample. For example,
          ``samples[0, :]`` is the parameter values for the first sample.
        * ``logprior``: 1-d `~numpy.ndarray` (length=nsamples);
          log(prior volume) for each sample.
        * ``logl``: 1-d `~numpy.ndarray` (length=nsamples); log(likelihood)
          for each sample.
        * ``weights``: 1-d `~numpy.ndarray` (length=nsamples);
          Weight corresponding to each sample. The weight is proportional to
          the prior * likelihood for the sample.
        * ``parameters``: 1-d `~numpy.ndarray` of weighted-mean parameter
          values from samples (including fixed parameters). Order corresponds
          to ``model.param_names``.
        * ``covariance``: 2-d `~numpy.ndarray` of parameter covariance;
          indicies correspond to order of ``vparam_names``. Calculated from
          ``samples`` and ``weights``.
        * ``errors``: OrderedDict of varied parameter uncertainties.
          Corresponds to square root of diagonal entries in covariance matrix.
        * ``ndof``: Number of degrees of freedom (len(data) -
          len(vparam_names)).
        * ``bounds``: Dictionary of bounds on varied parameters (including
          any automatically determined bounds).

    estimated_model : `~sncosmo.Model`
        A copy of the model with parameters set to the values in
        ``res.parameters``.

    Notes
    -----

    The algorithm uses the numpy random number generator to generate samples,
    and is therefore non-deterministic in default use. To get reproducible
    results, simply seed the random number generator before calling `nest_lc`:

    >>> import numpy as np
    >>> np.random.seed(0)

    """

    data = standardize_data(data)
    model = copy.copy(model)
    bounds = copy.copy(bounds)  # need to copy this dict b/c we modify it below

    # Order vparam_names the same way it is ordered in the model:
    vparam_names = [s for s in model.param_names if s in vparam_names]

    # Drop data that the model doesn't cover.
    data = cut_bands(data, model, z_bounds=bounds.get('z', None))

    if guess_amplitude_bound:
        if model.param_names[2] not in vparam_names:
            raise ValueError("Amplitude bounds guessing enabled but "
                             "amplitude parameter {0!r} is not varied"
                             .format(model.param_names[2]))
        if model.param_names[2] in bounds:
            raise ValueError("cannot supply bounds for parameter {0!r}"
                             " when guess_amplitude_bound=True"
                             .format(model.param_names[2]))

        # If redshift is bounded, set model redshift to midpoint of bounds
        # when doing the guess.
        if 'z' in bounds:
            model.set(z=sum(bounds['z']) / 2.)
        _, amplitude = guess_t0_and_amplitude(data, model, minsnr)
        bounds[model.param_names[2]] = (0., 10. * amplitude)

    # Find t0 bounds to use, if not explicitly given
    if 't0' in vparam_names and 't0' not in bounds:
        bounds['t0'] = t0_bounds(data, model)

    res = _nest_lc(data, model, vparam_names, modelcov=modelcov,
                   bounds=bounds, priors=priors, ppfs=ppfs, nobj=nobj,
                   maxiter=maxiter, maxcall=maxcall, verbose=verbose)

    res.bounds = bounds

    # calculate weighted average and weighted covariance matrix of samples
    vparameters, cov = weightedcov(res['samples'], res['weights'])

    model.set(**dict(zip(vparam_names, vparameters)))
    res.parameters = model.parameters.copy()
    res.covariance = cov
    res.errors = odict(zip(vparam_names, np.sqrt(np.diagonal(cov))))
    res.param_dict = odict(zip(model.param_names, model.parameters))

    # TODO remove/change in a future release.
    depmsg = ("The `param_names` attribute is deprecated in sncosmo v1.0 "
              "and will be changed in v1.1. Use `vparam_names` instead.")
    res.__dict__['deprecated']['param_names'] = (res.vparam_names, depmsg)

    return res, model


def mcmc_lc(data, model, vparam_names, bounds=None, priors=None,
            guess_amplitude=True, guess_t0=True, guess_z=True,
            minsnr=5., modelcov=False, nwalkers=10, nburn=200,
            nsamples=1000, thin=1, a=2.0):
    """Run an MCMC chain to get model parameter samples.

    This is a convenience function around `emcee.EnsembleSampler`.
    It defines the likelihood function and makes a heuristic guess at a
    good set of starting points for the walkers. It then runs the sampler,
    starting with a burn-in run.

    If you're not getting good results, you might want to try
    increasing the burn-in, increasing the walkers, or specifying a
    better starting position.  To get a better starting position, you
    could first run `~sncosmo.fit_lc`, then run this function with all
    ``guess_[name]`` keyword arguments set to False, so that the
    current model parameters are used as the starting point.

    Parameters
    ----------
    data : `~astropy.table.Table` or `~numpy.ndarray` or `dict`
        Table of photometric data. Must include certain columns.
        See the "Photometric Data" section of the documentation for
        required columns.
    model : `~sncosmo.Model`
        The model to fit.
    vparam_names : iterable
        Model parameters to vary.
    bounds : `dict`, optional
        Bounded range for each parameter. Keys should be parameter
        names, values are tuples. If a bound is not given for some
        parameter, the parameter is unbounded. The exception is
        ``t0``: by default, the minimum bound is such that the latest
        phase of the model lines up with the earliest data point and
        the maximum bound is such that the earliest phase of the model
        lines up with the latest data point.
    priors : `dict`, optional
        Prior probability functions. Keys are parameter names, values are
        functions that return probability given the parameter value.
        The default prior is a flat distribution.
    guess_amplitude : bool, optional
        Whether or not to guess the amplitude from the data. If false, the
        current model amplitude is taken as the initial value. Only has an
        effect when fitting amplitude. Default is True.
    guess_t0 : bool, optional
        Whether or not to guess t0. Only has an effect when fitting t0.
        Default is True.
    guess_z : bool, optional
        Whether or not to guess z (redshift). Only has an effect when fitting
        redshift. Default is True.
    minsnr : float, optional
        When guessing amplitude and t0, only use data with signal-to-noise
        ratio (flux / fluxerr) greater than this value. Default is 5.
    modelcov : bool, optional
        Include model covariance when calculating chisq. Default is False.
    nwalkers : int, optional
        Number of walkers in the EnsembleSampler
    nburn : int, optional
        Number of samples in burn-in phase.
    nsamples : int, optional
        Number of samples in production run.
    thin : int, optional
        Factor by which to thin samples in production run. Output samples
        array will have (nsamples/thin) samples.
    a : float, optional
        Proposal scale parameter passed to the EnsembleSampler.

    Returns
    -------
    res : Result
        Has the following attributes:

        * ``param_names``: All parameter names of model, including fixed.
        * ``parameters``: Model parameters, with varied parameters set to
          mean value in samples.
        * ``vparam_names``: Names of parameters varied. Order of parameters
          matches order of samples.
        * ``samples``: 2-d array with shape ``(N, len(vparam_names))``.
          Order of parameters in each row  matches order in
          ``res.vparam_names``.
        * ``covariance``: 2-d array giving covariance, measured from samples.
          Order corresponds to ``res.vparam_names``.
        * ``errors``: dictionary giving square root of diagonal of covariance
          matrix for varied parameters. Useful for ``plot_lc``.
        * ``mean_acceptance_fraction``: mean acceptance fraction for all
          walkers in the sampler.

    est_model : `~sncosmo.Model`
        Copy of input model with varied parameters set to mean value in
        samples.

    """

    try:
        import emcee
    except:
        raise ImportError("mcmc_lc() requires the emcee package.")

    # Standardize and normalize data.
    data = standardize_data(data)
    data = normalize_data(data)

    # Make a copy of the model so we can modify it with impunity.
    model = copy.copy(model)

    if bounds is None:
        bounds = {}
    if priors is None:
        priors = {}

    # Check that vparam_names isn't empty, check for unknown parameters.
    if len(vparam_names) == 0:
        raise ValueError("no parameters supplied")
    for names in (vparam_names, bounds, priors):
        for name in names:
            if name not in model.param_names:
                raise ValueError("Parameter not in model: " + repr(name))

    # Order vparam_names the same way it is ordered in the model:
    vparam_names = [s for s in model.param_names if s in vparam_names]
    ndim = len(vparam_names)

    # Check that 'z' is bounded (if it is going to be fit).
    if 'z' in vparam_names:
        if 'z' not in bounds or None in bounds['z']:
            raise ValueError('z must be bounded if allowed to vary.')
        if guess_z:
            model.set(z=sum(bounds['z']) / 2.)
        if model.get('z') < bounds['z'][0] or model.get('z') > bounds['z'][1]:
            raise ValueError('z out of range.')

    # Cut bands that are not allowed by the wavelength range of the model.
    data = cut_bands(data, model, z_bounds=bounds.get('z', None))

    # Find t0 bounds to use, if not explicitly given
    if 't0' in vparam_names and 't0' not in bounds:
        bounds['t0'] = t0_bounds(data, model)

    # Note that in the parameter guessing below, we assume that the source
    # amplitude is the 3rd parameter of the Model (1st parameter of the Source)

    # Turn off guessing if we're not fitting the parameter.
    if model.param_names[2] not in vparam_names:
        guess_amplitude = False
    if 't0' not in vparam_names:
        guess_t0 = False

    # Make guesses for t0 and amplitude.
    # (we assume amplitude is the 3rd parameter of the model.)
    if guess_amplitude or guess_t0:
        t0, amplitude = guess_t0_and_amplitude(data, model, minsnr)
        if guess_amplitude:
            model.parameters[2] = amplitude
        if guess_t0:
            model.set(t0=t0)

    # Indicies used in probability function.
    # modelidx: Indicies of model parameters corresponding to vparam_names.
    # idxbounds: tuples of (varied parameter index, low bound, high bound).
    # idxpriors: tuples of (varied parameter index, function).
    modelidx = np.array([model.param_names.index(k) for k in vparam_names])
    idxbounds = [(vparam_names.index(k), bounds[k][0], bounds[k][1])
                 for k in bounds]
    idxpriors = [(vparam_names.index(k), priors[k]) for k in priors]

    # Posterior function.
    def lnprob(parameters):
        for i, low, high in idxbounds:
            if not low < parameters[i] < high:
                return -np.inf

        model.parameters[modelidx] = parameters
        logp = -0.5 * _chisq(data, model, modelcov=modelcov)

        for i, func in idxpriors:
            logp += math.log(func(parameters[i]))

        return logp

    # Heuristic determination of  walker initial positions:
    # distribute walkers in a symmetric gaussian ball, with heuristically
    # determined scale.
    ctr = model.parameters[modelidx]
    scale = np.ones(ndim)
    for i, name in enumerate(vparam_names):
        if name in bounds:
            scale[i] = 0.0001 * (bounds[name][1] - bounds[name][0])
        elif model.get(name) != 0.:
            scale[i] = 0.01 * model.get(name)
        else:
            scale[i] = 0.1
    pos = ctr + scale * np.random.normal(size=(nwalkers, ndim))

    # Run the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a=a)
    pos, prob, state = sampler.run_mcmc(pos, nburn)  # burn-in
    sampler.reset()
    sampler.run_mcmc(pos, nsamples, thin=thin)  # production run
    samples = sampler.flatchain

    # Summary statistics.
    vparameters = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=0)
    model.set(**dict(zip(vparam_names, vparameters)))
    errors = odict(zip(vparam_names, np.sqrt(np.diagonal(cov))))
    mean_acceptance_fraction = np.mean(sampler.acceptance_fraction)

    res = Result(param_names=copy.copy(model.param_names),
                 parameters=model.parameters.copy(),
                 vparam_names=vparam_names,
                 samples=samples,
                 covariance=cov,
                 errors=errors,
                 mean_acceptance_fraction=mean_acceptance_fraction)

    return res, model
