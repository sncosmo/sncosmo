from sys import stdout
import time
import math
import copy

import numpy as np
from scipy import integrate, optimize
from astropy.utils import OrderedDict as odict

from .models import get_model
from .spectral import get_magsystem
from .photometric_data import standardize_data, normalize_data
from . import nest

def _cdf(pdf, x, a):
    return integrate.quad(pdf, a, x)[0]

def _ppf_to_solve(x, pdf, q, a):
    return _cdf(pdf, x, a) - q

def _ppf_single_call(pdf, q, a, b):
    left = right = None
    if a > -np.inf: left = a
    if b < np.inf: right = b

    factor = 10.

    # if lower limit is -infinity, adjust to
    # ensure that cdf(left) < q
    if  left is None:
        left = -1. * factor
        while _cdf(pdf, left, a) > q:
            right = left
            left *= factor

    # if upper limit is infinity, adjust to
    # ensure that cdf(right) > q
    if  right is None:
        right = factor
        while _cdf(pdf, right, a) < q:
            left = right
            right *= factor

    return optimize.brentq(_ppf_to_solve, left, right, args=(pdf, q, a))

class Interp1d(object):
    def __init__(self, xmin, xmax, y):
        self._xmin = xmin
        self._xmax = xmax
        self._n = len(y)
        self._xstep = (xmax - xmin) / (self._n - 1)
        self._y = y

    def __call__(self, x):
        """works only in range [xmin, xmax)"""
        nsteps = (x - self._xmin) / self._xstep
        i = int(nsteps)
        w = nsteps - i
        return (1.-w) * self._y[i] + w * self._y[i+1]

def pdf_to_ppf(pdf, a, b):
    """Given a function representing a pdf, return a callable representing the
    inverse cdf (or ppf) of the pdf."""

    n = 101
    x = np.linspace(0., 1., n)
    y = np.empty(n, dtype=np.float)
    y[0] = a
    y[-1] = b
    for i in range(1, n-1):
        y[i] = _ppf_single_call(pdf, x[i], a, b)

    return Interp1d(0., 1., y)


def evidence(model, data, parnames,
             parlims=None, priors=None, ppfs=None, tied=None,
             include_error=False, nobj=50, maxiter=10000,
             verbose=False, verbose_name=''):

    # Construct a list of ppfs to be used in the prior() function...
    npar = len(parnames)
    ppflist = npar * [None]

    # ... if a ppf is directly supplied for a parameter, it takes precedence...
    if ppfs is not None:
        for i, parname in enumerate(parnames):
            if parname in ppfs:
                ppflist[i] = ppfs[parname]

    # ...and for the parameters without ppfs, construct one from limits/prior.
    for i, parname in enumerate(parnames):
        if ppflist[i] is not None:
            continue
        if parname not in parlims:
            raise ValueError("Must supply ppf or limits for parameter '{}'"
                             .format(parname))
        a, b = parlims[parname]
        if (priors is not None and parname in priors):
            ppflist[i] = pdf_to_ppf(priors[parname], a, b)
        else:
            ppflist[i] = Interp1d(0., 1., np.array([a, b]))

    def prior(u):
        v = np.empty(npar, dtype=np.float)
        for i in range(npar):
            v[i] = ppflist[i](u[i])
        return v

    def loglikelihood(parvals):
        d = dict(zip(parnames, parvals))
        if tied is not None:
            for parname, func in tied.iteritems():
                d[parname] = func(d)

        model.set(**d)
        if not include_error:
            mflux = model.bandflux(data['band'], data['time'],
                                       zp=data['zp'], zpsys=data['zpsys'])
            chisq = np.sum(((data['flux'] - mflux) / data['fluxerr'])**2)
        else:
            mflux, mfluxerr =  model.bandflux(
                data['band'], data['time'], zp=data['zp'], zpsys=data['zpsys'],
                include_error=True)
            chisq = np.sum(((data['flux'] - mflux)**2 /
                            (data['fluxerr']**2 + mfluxerr**2)))

        return -chisq / 2.

    res = nest.nest(loglikelihood, prior, npar, nobj=nobj, maxiter=maxiter,
                    verbose=verbose, verbose_name=verbose_name)

    # Add tied parameters to results. This is inelegant, but, eh.
    nsamples = len(res['samples_parvals'])
    res['nsamples'] = nsamples
    if tied is not None:
        tiedparnames = tied.keys()
        ntiedpar = len(tiedparnames)
        tiedparvals = np.empty((nsamples, ntiedpar), dtype=np.float)
        for i in range(nsamples):
            d = dict(zip(parnames, res['samples_parvals'][i, :]))
            for j, parname in enumerate(tiedparnames):
                tiedparvals[i, j] = tied[parname](d)

        res['samples_parvals'] = np.hstack((res['samples_parvals'], 
                                            tiedparvals))
        parnames = parnames + tiedparnames

    # Sample averages and their standard deviations.
    res['parvals'] = np.average(res['samples_parvals'],
                                weights=res['samples_wt'], axis=0)
    res['parerrs'] = np.sqrt(np.sum(res['samples_wt'][:, np.newaxis] *
                             res['samples_parvals']**2, axis=0) -
                             res['parvals']**2)

    # Add some more to results
    res['parnames'] = parnames
    res['chisq_min'] = -2. * res.pop('loglmax')
    res['dof'] = len(time) - npar

    return res


class PhotoTyper(object):
    """Baysian photometric typer.
    
    Parameters
    ----------
    verbose : bool, optional
        Print lines as evidence is calculated.
    t0_range : tuple of floats: (t0_low, t0_high), optional
        Lower limit on t0 relative to earliest data point, and upper limit
        on t0 relative to latest data point, in days. Default is (0., 0.).

    Notes
    -----
    Parameters can also be set after initialization with, e.g.,
    ``typer.verbose = True`` or ``typer.t0_range = (-10., 10.)``.
    """

    def __init__(self, verbose=True, t0_range=(-10., 10.)):
        self._models = odict()
        self.types = []
        self.verbose = verbose
        self.t0_range = t0_range

    def add_model(self, model, model_type, parlims, priors=None,
                  model_prior=1., tied=None, include_error=False,
                  name=None):
        """Add a model.

        Parameters
        ----------
        model : `~sncosmo.Model` or str
            A model instance or name of a model in the registry. 
        model_type : str
            A string identifier for the type of the model (e.g., 'SN Ia',
            'SN IIP', etc). Models of the same type are included together
            in determining probabilities.
        parlims : dict, optional
            Dictionary.
        tied : dict, optional
            Dictionary of functions, default is `None`.
        name : str
            Name 
        """

        model = get_model(model)  # This always copies the model.
        if name is None:
            name = model.name
        if name is None:
            raise ValueError('Model does not have a name. Set the name.')
        if name in self._models:
            raise ValueError('model of this name already included.')
        if model_prior <= 0.:
            raise ValueError('model_prior must be positive')

        # get ppf for each parameter
        ppfs = {}
        for parname, parlim in parlims.iteritems():
            a, b = parlims[parname]
            if (priors is not None and parname in priors):
                ppfs[parname] = pdf_to_ppf(priors[parname], a, b)
            else:
                ppfs[parname] = Interp1d(0., 1., np.array([a, b]))

        # Add model info to internal data.
        self._models[name] = {
            'model': model,
            'type': model_type,
            'parlims': parlims,
            'ppfs': ppfs,
            'tied': tied,
            'model_prior': model_prior,
            'include_error': include_error
            }

        # Add model type to list of types.
        if model_type not in self.types:
            self.types.append(model_type)

    def __str__(self):
        lines = []
        for model_type in self.types:
            lines.append('Type: {}'.format(model_type))
            for name, d in self._models.iteritems():
                if d['type'] != model_type: continue
                lines.append('  Model: {}'.format(name))
                for parname, parvals in d['parlims'].iteritems():
                    lines.append('    {}: [{} .. {}]'
                                 .format(parname, parvals[0], parvals[1]))
        return '\n'.join(lines)

    def classify(self, data, verbose=None):
        """Determine probability of each model type for the given data.

        Parameters
        ----------
        data : `~numpy.ndarray` or dict
            Light curve data to classify.
        verbose : bool, optional
            If True, print information during iteration. (If False, don't).
            Default is to use the value of self.verbose.

        Returns
        -------
        types : OrderedDict
            Keys are type names ('SN Ia'), values are dictionaries
            containing key 'p'. Entries are sorted by 'p', decreasing.
        models : OrderedDict
            Keys are model names. Values are dictionaries with the
            following keys:
           
            * 'p' : probability (float)
            * 'perr' : numerical error on p (tuple)
            * 'type' : Type of model (str)
            * 'dof' : degrees of freedom (len(data) - npar)
            * 'niter' : number of iterations (int)
            * 'ncalls' : number of likelihood calls (int)
            * 'time' : evaluation time in seconds (float)
            * 'parnames': parameter names (list of str)
            * 'parvals': parameter "best-fit" values (list of float)
            * 'parerrs': parameter "errors" (list of float)
            * 'chisq_min': minimum chi^2 value of any sample (float)
            * 'samples_parvals' : ndarray, shape=(nsamples, npars)
            * 'samples_wt': ndarray, shape=(nsamples,)

            Entries are sorted by 'p', decreasing.
        """

        if verbose is None:
            verbose = self.verbose

        # Initialize data
        data = standardize_data(data)

        # limit data to bands that overlap *all* models over the full z range.
        valid = np.ones(len(data), dtype=np.bool)
        for m in self._models.values():
            model = m['model']
            if 'z' not in m['parlims']:
                v = model.bandoverlap(data['band'])
            else:
                v = np.all(model.bandoverlap(data['band'],z=m['parlims']['z']),
                           axis=1)
            valid = valid & v
        if not np.all(valid):
            drop_bands = [repr(b) for b in set(data['band'][np.invert(valid)])]
            warn("Dropping following bands from data: " +
                 ", ".join(drop_bands) + "(out of model wavelength range)",
                 RuntimeWarning)
            data = data[valid]

        # get range of t0 to consider
        parlims = {
            't0': (np.min(data['time']) + self.t0_range[0],
                   np.max(data['time']) + self.t0_range[1])
            }

        models = {}
        for name, m in self._models.iteritems():
            parnames = m['ppfs'].keys() + ['t0']
            models[name] = evidence(
                m['model'], data, parnames,
                parlims=parlims, ppfs=m['ppfs'], tied=m['tied'],
                include_error=m['include_error'],
                verbose=verbose, verbose_name=name)

            # multiply evidence by model prior
            models[name]['logz'] += m['model_prior']

            # add type info
            models[name]['type'] = m['type']

        # get denominator (sum of Z)
        logzvals = [d['logz'] for d in models.values()]
        logzsum = logzvals[0]
        for i in range(1, len(logzvals)):
            logzsum = np.logaddexp(logzsum, logzvals[i])
        
        # get probability of each model: p = Z_i / sum(Z)
        # Errors are calculated by finding the upper and lower Z limits:
        # up   = exp(logz + logzerr) = exp(logz)exp(logzerr) = z*exp(logzerr)
        # down = exp(logz - logzerr) = exp(logz)/exp(logzerr) = z/exp(logzerr)
        for name, d in models.iteritems():
            p = np.exp(d['logz'] - logzsum)
            explogzerr = math.exp(d['logzerr'])
            p_up = p * explogzerr / (1. - p + p * explogzerr)
            p_dn = p / explogzerr / (1. - p + p / explogzerr)
            d['p'] = p
            d['perr'] = (p_up - p, p - p_dn)

        # get probability for each type
        types = {name: {'p': 0.} for name in self.types}
        for name, d in models.iteritems():
            types[d['type']]['p'] += d['p']

        # sort models and types
        types = odict(
            sorted(types.items(), key=lambda t: t[1]['p'], reverse=True)
            )
        models = odict(
            sorted(models.items(), key=lambda t: t[1]['p'], reverse=True)
            )

        return types, models
