import math
from warnings import warn
from operator import mul

import numpy as np
from astropy.utils import OrderedDict as odict

from .models import _ModelBase, ObsModel
from .spectral import get_magsystem
from .photdata import standardize_data
from .fitting import _nest_lc
from .utils import pdf_to_ppf, Interp1d


class PhotoTyper(object):
    """Bayesian photometric typer.
    
    Parameters
    ----------
    verbose : bool, optional
        Print lines as evidence is calculated.
    t0_extend : float, optional
        Extend t0 bounds this far before and after earliest and latest data
        point.

    Notes
    -----
    Parameters can also be set after initialization with, e.g.,
    ``typer.verbose = True`` or ``typer.t0_extend = 10.``.
    """

    def __init__(self, verbose=True, t0_extend=0.):
        self._models = odict()
        self.types = []
        self.verbose = verbose
        self.t0_extend = t0_extend

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

        #model = get_model(model)  # This always copies the model.
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
            lines.append('Type: {0}'.format(model_type))
            for name, d in self._models.iteritems():
                if d['type'] != model_type: continue
                lines.append('  Model: {0}'.format(name))
                for parname, parvals in d['parlims'].iteritems():
                    lines.append('    {0}: [{1} .. {2}]'
                                 .format(parname, parvals[0], parvals[1]))
        return '\n'.join(lines)

    def classify(self, data, t0_bounds=None, verbose=None):
        """Determine probability of each model type for the given data.

        Parameters
        ----------
        data : `~numpy.ndarray` or dict
            Light curve data to classify.
        t0_bounds : (float, float), optional
            If given, use this range as a flat prior on t0 for all models.
            If not given, default is to use the minimum and maximum time
            in the data as the bounds.
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

        # set the range of t0 values
        if t0_bounds is None:
            t0_bounds = (np.min(data['time']) - self.t0_extend,
                         np.max(data['time']) + self.t0_extend)
        parlims = {'t0': t0_bounds}

        models = {}
        for name, m in self._models.iteritems():
            parnames = m['ppfs'].keys() + ['t0']
            models[name] = _nest_lc(
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
        types = dict([(name, {'p': 0.}) for name in self.types])
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
