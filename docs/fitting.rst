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
