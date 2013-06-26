from sys import stdout
import time
import math
import copy
import operator

import numpy as np
from scipy import integrate, optimize
from astropy.utils import OrderedDict

from .models import get_model
from .spectral import get_magsystem
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
             return_samples=False, verbose=False, verbose_name=''):

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

    #def prior(u):
    #    return parlims[:, 0] + u * (parlims[:, 1] - parlims[:, 0])

    band = data['band']
    time = data['time']
    zp = data['zp']
    zpsys = data['zpsys']
    flux = data['flux']
    fluxerr = data['fluxerr']

    def loglikelihood(parvals):
        d = dict(zip(parnames, parvals))
        if tied is not None:
            for parname, func in tied.iteritems():
                d[parname] = func(d)

        model.set(**d)
        if not include_error:
            modelflux = model.bandflux(band, time, zp, zpsys)
            chisq = np.sum(((flux - modelflux) / fluxerr)**2)
        else:
            modelflux, modelfluxerr =  model.bandflux(band, time, zp, zpsys,
                                                      include_error=True)
            chisq = np.sum(((flux - modelflux)**2 /
                            (fluxerr**2 + modelfluxerr**2)))

        return -chisq / 2.

    res = nest.nest(loglikelihood, prior, npar, nobj=nobj, maxiter=maxiter,
                    return_samples=return_samples, verbose=verbose,
                    verbose_name=verbose_name)
    res['parnames'] = parnames
    res['chisq_min'] = -2. * res.pop('loglmax')
    res['dof'] = len(time) - npar
    return res


class PhotoTyper(object):
    """Baysian photometric typer.
    """

    def __init__(self, verbose=True):
        self._models = OrderedDict()
        self.types = []
        self.verbose = verbose

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
        parlims : dict
            Dictionary.
        tied : dict
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

    def classify(self, data, return_samples=False, verbose=None):
        """Determine probability of each model type for the given data.

        Parameters
        ----------
        data : `~numpy.ndarray` or dict
            Light curve data to classify.
        return_samples : bool, optional
            If True, add samples to `bestmodel_params`
        verbose : bool, optional
            If True, print information during iteration. (If False, don't).
            Default is to use the value of self.verbose.

        Returns
        -------
        type_p : dict
            Probability for each model type.
        model_p : dict
            Probability for each model.
        model_perr : dict of tuples: (uperr, downerr)
            Approximate computational probability error for each model.
        bestmodel : str
            Name of model with highest probability.
        bestmodel_params : dict
            Model parameters and uncertainties for highest-probability model,
            with keys: 'parnames', 'parvals', 'parerrs' (each is a list).
        """

        if verbose is None:
            verbose = self.verbose

        # limit data to bands that overlap *all* models over the full z range.
        valid = np.ones(len(data['band']), dtype=np.bool)
        for m in self._models.values():
            model = m['model']
            if 'z' not in m['parlims']:
                v = model.bandoverlap(data['band'])
            else:
                v = np.all(model.bandoverlap(data['band'],z=m['parlims']['z']),
                           axis=1)
            valid = valid & v
        if not np.all(valid):
            print "WARNING: dropping following bands from data:"
            print np.unique(data['band'][np.invert(valid)])
            data = {'time': data['time'][valid],
                    'band': data['band'][valid],
                    'flux': data['flux'][valid],
                    'fluxerr': data['fluxerr'][valid],
                    'zp': data['zp'][valid],
                    'zpsys': data['zpsys'][valid]}

        # get range of t0 to consider
        parlims = {'t0': (np.min(data['time']), np.max(data['time']))}

        logz = {}  # Log evidence for each model
        logzerr = {}
        model_params = {}
        for name, m in self._models.iteritems():
            parnames = m['ppfs'].keys() + ['t0']
            res = evidence(m['model'], data, parnames,
                           parlims=parlims, ppfs=m['ppfs'], tied=m['tied'],
                           include_error=m['include_error'],
                           verbose=verbose, verbose_name=name,
                           return_samples=return_samples)

            # accumulate info
            logz[name] = res['logz'] + m['model_prior']
            logzerr[name] = res['logzerr']
            model_params[name] = {'parnames': res['parnames'],
                                  'parvals': res['parvals'],
                                  'parerrs': res['parerrs'],
                                  'chisq_min': res['chisq_min'],
                                  'dof': res['dof']}
            if return_samples:
                model_params[name]['samples_parvals'] = res['samples_parvals']
                model_params[name]['samples_wt'] = res['samples_wt']

        # get denominator (sum of Z)
        logzvals = logz.values()
        logzsum = logzvals[0]
        for i in range(1, len(logzvals)):
            logzsum = np.logaddexp(logzsum, logzvals[i])
        
        # get probability of each model: p = Z_i / sum(Z)
        model_p = {}
        for name, val in logz.iteritems():
            model_p[name] = np.exp(logz[name] - logzsum)
        
        # get error for each model:
        # up   = exp(logz + logzerr) = exp(logz)exp(logzerr) = z*exp(logzerr)
        # down = exp(logz - logzerr) = exp(logz)/exp(logzerr) = z/exp(logzerr)
        model_perr = {}
        for name, val in model_p.iteritems():
            up = (model_p[name] * math.exp(logzerr[name]) /
                  (1. + model_p[name] * (math.exp(logzerr[name]) - 1.)))
            down = (model_p[name] / math.exp(logzerr[name]) /
                    (1.-model_p[name] + model_p[name]/math.exp(logzerr[name])))
            model_perr[name] = (up - model_p[name], model_p[name] - down)

        # get probability for each type
        type_p = {name:0. for name in self.types}
        for name, m in self._models.iteritems():
            type_p[m['type']] += model_p[name]

        # find highest probability model
        bestmodel = max(model_p.iteritems(), key=operator.itemgetter(1))[0]

        return type_p, model_p, model_perr, bestmodel, model_params[bestmodel]
