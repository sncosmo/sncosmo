from sys import stdout
import numpy as np
import math

from .models import get_model

__all__ = ['nested_sampling']

# Code from scipy for getting inverse cdf (ppf) for when priors are used.
# 
# from scipy/stats/distributions.py
"""
def _cdf(pdf, x, a):
    return integrate.quad(pdf, a, x)[0]

def _ppf_to_solve(pdf, x, q, a)
    return _cdf(pdf, x, a) - q

def _ppf_single_call(pdf, q, a, b):
    left = right = None
    if a > -np.inf: left = a
    if b < np.inf: right = b

    factor = 10.
    if  left is None:
        left = -1. * factor
        while self._cdf(pdf, left, a) > q:
            right = left
            left *= factor
        # left is now such that cdf(left) < q
    if  right is None: # i.e. self.b = inf
        right = factor
        while self._cdf(pdf, right, a) < q:
            left = right
            right *= factor
        # right is now such that cdf(right) > q

    return optimize.brentq(_ppf_to_solve, \
                           left, right, args=(q,)+args, xtol=self.xtol)
"""

# For now doesn't include model errors
def _loglikelihood(model, data):
    modelflux = model.bandflux(data['band'], data['time'],
                               zp=data['zp'], zpsys=data['zpsys'])
    chisq = np.sum(((data['flux'] - modelflux) / data['fluxerr'])**2)
    return -chisq / 2.

def nested_sampling(model, data, parlims, tied=None, maxiter=1000):
    """Evaluate the model evidence and parameter given the data by nested
    sampling. Currently uses a uniform prior between the parameter limits.

    Parameters
    ----------
    model : 
    data :
    parlims : dict
        Keys are parameters to vary, values are limits: ``(low, high)``.
    tied : dict
        Keys are tied parameter names, values are functions accepting
        a dictionary of parameter values and returning value of the tied
        parameter.
    
    Returns
    -------
    niter : int
        Number of iterations
    params : dict
        Parameter values and standard deviations
    logz : tuple
        Natural log of evidence ``Z`` and its uncertainty
    h : tuple
        Information ``H`` and its uncertainty.

    Notes
    -----
    This is an implementation of John Skilling's Nested Sampling algorithm.
    More information: http://www.inference.phy.cam.ac.uk/bayesys/
    """

    parnames = parlims.keys()
    parlims = np.array(parlims.values())
    npar = len(parnames)

    # values for now
    nexplore = 20
    nobj = 100

    # select objects from the prior
    dt = [('u',np.float,(npar,)), ('v',np.float,(npar,)), ('logl',np.float)]
    objects = np.empty(nobj, dtype=dt)
    objects['u'] = np.random.random((nobj, npar))
    objects['v'] = parlims[:, 0] + objects['u'] * (parlims[:, 1]-parlims[:, 0])
    for i in range(nobj):
        d = dict(zip(parnames, objects['v'][i]))
        if tied is not None:
            for parname, func in tied.iteritems():
                d[parname] = func(d)
        model.set(**d)
        objects['logl'][i] = _loglikelihood(model, data)

    # Initialize values for nested sampling loop.
    samples = []  # Objects stored for posterior results.
    loglstar = None  # ln(Likelihood constraint)
    h = 0.  # Information, initially 0.
    logz = -1.e300  # ln(Evidence Z, initially 0)
    # ln(width in prior mass), outermost width is 1 - e^(-1/n)
    logwidth = math.log(1. - math.exp(-1./nobj))

    # Nested sampling loop.
    print "model = {} [{:7d}/{:7d}]".format(model.name, 0, maxiter),
    for nest in range(maxiter):
        print 16 * '\b' + '{:7d}'.format(nest),
        stdout.flush()


        # worst object in collection and its weight (= width * likelihood)
        worst = np.argmin(objects['logl'])
        logwt = logwidth + objects['logl'][worst]

        # update evidence Z and information h.
        logz_new = np.logaddexp(logz, logwt)
        h = (math.exp(logwt - logz_new) * objects['logl'][worst] +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new

        # Add worst object to samples.
        samples.append({'v': np.array(objects['v'][worst]), 'logwt': logwt})

        # The new likelihood constraint is that of the worst object.
        loglstar = objects['logl'][worst]

        # Replace the worst object with a copy of a different object.
        tocopy = np.random.randint(nobj)
        while tocopy == worst:
            tocopy = np.random.randint(nobj)
        objects[worst] = objects[tocopy]

        # Explore space around copied object
        step = 0.1  # Initial guess suitable step-size in (0,1)
        accept = 0  # MCMC acceptances
        reject = 0  # MCMC rejections
        for m in range(nexplore):  # pre-judged number of steps
            u = objects['u'][worst] + step * (2.*np.random.random(npar)-1.)
            u -= np.floor(u)

            # Parameter values and likelihood.
            v = parlims[:, 0] + u * (parlims[:, 1] - parlims[:, 0])
            d = dict(zip(parnames, v))
            if tied is not None:
                for parname, func in tied.iteritems():
                    d[parname] = func(d)
            model.set(**d)
            logl = _loglikelihood(model, data)

            # Accept if and only if within likelihood constraint.
            if logl > loglstar:
                objects['u'][worst] = u
                objects['v'][worst] = v
                objects['logl'][worst] = logl
                accept += 1
            else:
                reject += 1

            # Refine step-size to let acceptance ratio converge around 50%
            if accept > reject:
                step *= math.exp(1./accept)
            elif accept < reject:
                step /= math.exp(1./reject)

        # Shrink interval
        logwidth -= 1./nobj

        # stopping condition goes here.

    # process samples and return
    niter = len(samples)
    samples_parvals = np.array([s['v'] for s in samples])  # (nsamp, npar)
    samples_logwt = np.array([s['logwt'] for s in samples])
    w = np.exp(samples_logwt - logz)  # Proportional weights.
    parvals = np.average(samples_parvals, axis=0, weights=w)
    parstds = np.sqrt(np.sum(w[:, np.newaxis] * samples_parvals**2, axis=0) -
                      parvals**2)
    params = dict(zip(parnames, zip(parvals, parstds)))
    logz_std = math.sqrt(h/nobj)
    h_std = h / math.log(2.)
    return niter, params, (logz, logz_std), (h, h_std)
    #return {'niter': niter,
    #        'parnames': parnames,
    #        'parvals': parvals,
    #        'parstds': parstds,
    #        'logz': logz,
    #        'logz_std': logz_std,
    #        'h': h,
    #        'h_std': h / math.log(2.)}

class PhotoTyper(object):
    """A set of models, each with a grid of parameters.
    """

    def __init__(self):
        self._models = OrderedDict()
        self.types = []

    def add_model(self, model, model_type, parlims, model_prior=None,
                  tied=None, name=None):
        """Add a model or models.

        Parameters
        ----------
        model : `~sncosmo.Model` or str
            A model instance or name of a model in the registry. 
        model_type : str
            A string identifier for the type of the model (e.g., 'SN Ia',
            'SN IIP', etc). Models of the same type are included together
            in determining probabilities.
        parvals : dict
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

        # Add model info to internal data.
        self._models[name] = {
            'model': model,
            'type': model_type,
            'tied': tied,
            'model_prior': model_prior,
            'parnames': parlims.keys(),
            'parlims': parlims.values()
            }

        # Precompute grid spacing for each parameter.
        #self._models[name]['dparvals'] = \
        #    [np.gradient(v) for v in self._models[name]['parvals']]

        # Add model to list of types.
        if model_type not in self.types:
            self.types.append(model_type)

    def __str__(self):
        lines = []
        for model_type in self.types:
            lines.append('Type: {}'.format(model_type))
            for name, d in self._models.iteritems():
                if d['type'] != model_type: continue
                ngridpoints = len(d['pargrid'])
                lines.append('  Model: {} [{} grid points]'
                             .format(name, ngridpoints))
                for parname, parvals in zip(d['parnames'], d['parvals']):
                    lines.append('    {}: [{} .. {}] {} values'.format(
                            parname, parvals[0], parvals[-1], len(parvals)))

        return '\n'.join(lines)

    def classify(self, data, verbose=False):
        """Determine probability of each model type for the given data.

        Parameters
        ----------
        data : 
        
        Returns
        -------
        probability : `dict`
            Dictionary giving the posterior probability for each model type.
        """

        # limit data to bands that overlap *all* models over the full z range.
        valid = np.ones(len(data['band']), dtype=np.bool)
        for m in self._models.values():
            model = m['model']
            if 'z' not in m['parnames']:
                valid = valid & model.bandoverlap(data['band'])
            else:
                i = m['parnames'].index('z')
                zbounds = [m['parvals'][i][0], m['parvals'][i][-1]]
                valid = (valid &
                         np.all(model.bandoverlap(data['band'],z=zbounds),
                                axis=1))
        if not np.all(valid):
            print "WARNING: dropping following bands from data:"
            print np.unique(data['band'][np.invert(valid)])
            data = {'time': data['time'][valid],
                    'band': data['band'][valid],
                    'flux': data['flux'][valid],
                    'fluxerr': data['fluxerr'][valid],
                    'zp': data['zp'][valid],
                    'zpsys': data['zpsys'][valid]}

        chisq = {}

        for name, m in self._models.iteritems():
            model = m['model']
            tied = m['tied']

            modelparams = model.params
            modelparnames = modelparams.keys() # parameter names in model
            d = {}
            # TODO: add t0 to param grid

            chisq[name] = np.empty(len(m['pargrid']), dtype=np.float)

            # Loop over parameters
            if verbose:
                print "model = {} [{:7d}/{:7d}]".format(
                    name, 0, len(m['pargrid'])),
            for i in range(len(m['pargrid'])):
                print 16 * '\b' + '{:7d}'.format(i),
                stdout.flush()

                params = dict(zip(m['parnames'], m['pargrid'][i, :]))

                # set the parameters that are in the model
                for parname, parvals in params.iteritems():
                    if parname in modelparnames:
                        d[parname] = parvals

                # Set dependent parameters
                for parname, func in tied.iteritems():
                    d[parname] = func(params)
                
                model.set(**d)
                modelflux = model.bandflux(
                    data['band'], data['time'],
                    zp=data['zp'], zpsys=data['zpsys'])
                chisq[name][i] = np.sum(((data['flux'] - modelflux) /
                                         data['fluxerr'])**2)
