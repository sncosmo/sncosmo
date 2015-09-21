*******************
Light Curve Fitting
*******************

*This is a user guide. For reference documentation, see the* :ref:`fitting-api`
*API*.

Photometric Data
================

See :doc:`photdata` for how to represent photometric data and how to
read/write it to various file formats. An important additional note: a
table of photometric data has a ``band`` column and a ``zpsys`` column
that use strings to identify the bandpass (e.g., ``'sdssg'``) and
zeropoint system (``'ab'``) of each observation. If the bandpass and
zeropoint systems in your data are *not* built-ins known to sncosmo,
you must register the corresponding `~sncosmo.Bandpass` or
`~sncosmo.MagSystem` to the right string identifier using the
:doc:`registry`.

Performing a fit
================

The following example fits a SALT2 model to some example data::

    >>> data = sncosmo.load_example_data()
    >>> model = sncosmo.Model(source='salt2')
    >>> res, fitted_model = sncosmo.fit_lc(data, model,
    ...                                    ['z', 't0', 'x0', 'x1', 'c'],
    ...                                    bounds={'z':(0.3, 0.7)})

Here, we are fitting the parameters of ``model`` to ``data``, and
we've supplied the names of the parameters that we want vary in the fit:
``['z', 't0', 'x0', 'x1', 'c']``. Since we're fitting redshift, we
need to provide bounds on the allowable redshift range. This is the
only parameter that *requires* bounds. Bounds are optional for the
other parameters.

The first object returned, ``res``, is a dictionary that has attribute
access (so ``res['params']`` and ``res.params`` both access the same thing).
You can see what attributes it has::

    >>> res.keys()
    ['errors', 'parameters', 'success', 'cov_names', 'covariance', 'ndof', 'chisq', 'param_names', 'message', 'ncall']

And then access those attributes::

    >>> res.ncall  # number of chi^2 function calls made
    132
    >>> res.ndof  # number of degrees of freedom in fit
    35
    >>> res.chisq  # chi^2 value at minimum
    33.50859337338642

The second object returned is a shallow copy of the input model with
the parameters set to the best fit values. The input model is
unchanged.

    >>> sncosmo.plot_lc(data, model=fitted_model, errors=res.errors)

.. image:: _static/example_lc.png


Fixing parameters (e.g., redshift) and setting initial guesses
==============================================================

Suppose we already know the redshift of the supernova we're trying to
fit.  We want to set the model's redshift to the known value, and then
make sure not to vary `z` in the fit::

    >>> model.set(z=0.5)  # set the model's redshift.
    >>> res, fitted_model = sncosmo.fit_lc(data, model,
    ...                                    ['t0', 'x0', 'x1', 'c'])


Using a custom fitter or sampler
================================

Besides `sncosmo.fit_lc`, sncosmo also provides two sampling methods
for estimating light curve parameters: `sncosmo.mcmc_lc` and
`sncosmo.nest_lc`. However, one may wish to experiment with a custom
fitting or sampling method. SNCosmo has been designed with this in
mind. In fact, the three supplied methods are fairly thin wrappers
around external python packages (respectively: iminuit, emcee, and
nestle).

For users wishing to use a custom fitting or sampling method, it can
be instructive to look at the source code for the built-in wrapper
functions. However, the basic idea is just that one can define a
likelihood for a set of light curve data given any sncosmo model, and
then that likelihood can be fed to any fitter or sampler.

How do you define the likelihood? Different fitters or samplers have
different requirements for the interface for the likelihood (or
posterior) function, so it will vary a bit. In the case where the
function must accept an array of parameter values, it can be defined
like this::

    def loglikelihood(parameters):
        model.parameters[:] = parameters  # set model parameters
        return -0.5 * sncosmo.chisq(data, model)

What does the `sncosmo.chisq` function do? The definition is basically::

    mflux = model.bandflux(data['band'], data['time'],
                           zp=data['zp'], zpsys=data['zpsys'])
    return np.sum(((data['flux'] - mflux) / data['fluxerr'])**2)

In other words, "use the model to predict the flux in the given
bandpasses and times (scaled with the appropriate zeropoint), then use
those values to calculate the chisq." So really `Model.bandflux` is
the key method that makes this all possible.

You might notice that our ``loglikelihood`` function above will vary *all*
the parameters in the model, which might not be what you want. To only vary
select parameters, you could do something like this::

    # Indicies of the model parameters that should be varied
    idx = np.array([model.param_names.index(name)
                    for name in param_names_to_vary])

    def loglikelihood(parameters):
        model.parameters[idx] = parameters
        return -0.5 * sncosmo.chisq(data, model)

The built-in wrapper functions in sncosmo don't do anything much more
complex than this. They take care of setting up the likelihood
function in the way that the underlying fitter or sampler
expects. They set guesses and bounds and package results up in a way
that is as consistent as possible across the three functions.
