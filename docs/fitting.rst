*******************
Light Curve Fitting
*******************

*Reference documentation is in* `sncosmo.fit_lc`.

First, have your photometric data in something that looks like a table, with a row for each observation. For what this can look like, see :doc:`photometric_data`. For now, get an example of photometric data::

    >>> meta, data = sncosmo.load_example_data()

Here, ``data`` is a structured `numpy.ndarray`, and ``meta`` is an `OrderedDict`. For a pretty-printed version of this, you can do::

    >>> from astropy.table import Table
    >>> print Table(data)
         time      band        flux          fluxerr      zp  zpsys
    ------------- ----- ----------------- -------------- ---- -----
          55070.0 sdssg    0.813499900062 0.651728140824 25.0    ab
    55072.0512821 sdssr  -0.0852238865812 0.651728140824 25.0    ab
    55074.1025641 sdssi -0.00681659003089 0.651728140824 25.0    ab
    55076.1538462 sdssz     2.23929135407 0.651728140824 25.0    ab
    55078.2051282 sdssg  -0.0308977349373 0.651728140824 25.0    ab
    55080.2564103 sdssr     2.35450321853 0.651728140824 25.0    ab
    ...

In this data, both the ``band`` column and the ``zpsys`` column
contain strings. These specific bands (e.g., ``'sdssg'``) and zeropoint
system (``'ab'``) are built-ins known to sncosmo, so we can feed these
to any sncosmo function and it will find the right corresponding
`~sncosmo.Bandpass` or `~sncosmo.MagSystem`. The data could also contain
the `~sncosmo.Bandpass` and `~sncosmo.MagSystem` objects themselves. Or, if it had strings that were not built-ins, you could register the corresponding `~sncosmo.Bandpass` or `~sncosmo.MagSystem` using the :doc:`registry`. 

Anyway, let's go ahead and fit a SALT2 model to the data::

    >>> model = sncosmo.get_model('salt2')
    <SALT2Model 'salt2' version='2.0' at 0x3a3c4d0>
    >>> res = sncosmo.fit_lc(data, model, ['mabs', 'x1', 'c', 't0', 'z'],
    ...                      bounds={'z':(0.3, 0.7)})

Here, we have said to fit the parameters of ``model`` to ``data``, and
we've supplied the parameters that we want to fit: ``['mabs', 'x1',
'c', 't0', 'z']``. Since we're fitting redshift, we need to provide
bounds on the allowable redshift range. This is the only parameter
that *requires* bounds. Bounds are optional for the other parameters.

The result, ``res`` is basically a dictionary with attribute
access (so ``res['params']`` and ``res.params`` both access the same thing).
You can see what attributes it has::

    >>> print res.keys()
    ['errors', 'params', 'matrix', 'covariance', 'ncalls', 'ndof', 'fval']

And then access those attributes::

    >>> print res.params  # best-fit values
    {'c': 0.1470013188713245, 'x1': 0.6050340491966963, 'z': 0.5397600044479324, 't0': 55100.34542300686, 'mabs': -19.735806129094037}
    >>> print res.ncalls  # number of function calls made
    238
    >>> print res.ndof  # number of degrees of freedom in fit
    35
    >>> print res.fval  # chi^2 value at minimum
    44.7949902843

To plot the fitted result, first set the parameters of the model to the best-fit values, then plot::

    >>> model.set(**res.params)
    >>> sncosmo.plot_lc(data, model)

.. image:: _static/example_lc.png


Fixing parameters (e.g., redshift) and setting initial guesses
==============================================================

Suppose we already know the redshift of the supernova we're trying to
fit.  We want to set the model's redshift to the known value, and then
make sure not to vary `z` in the fit::

    >>> model = sncosmo.get_model('salt2')
    >>> model.set(z=0.5)  # set the model's redshift.
    >>> res = sncosmo.fit_lc(data, model, ['mabs', 'x1', 'c', 't0']) # only vary 4 parameters.

Instead of the last two lines, we could also have simply done::

    >>> res = sncosmo.fit_lc(data, model, ['mabs', 'x1', 'c', 't0'], p0={'z': 0.5})

Any parameter values specified with the ``p0`` keyword argument will
override the current model parameters. ``p0`` can also be used to set
initial guesses for parameters that *are* varied in the fit. It will
override the default guesses that are made for ``t0`` and ``fscale``
parameters.


Fitting an "offset" for each bandpass
=====================================

It is possible to fit for a constant "offset" flux in each band. This
is useful in cases such as:

1. The photometry includes an unknown (but constant) amount of host galaxy
   light (e.g., the photometry was not done on "subtracted" images).

2. There is an unknown (but constant) amount of SN light in the
   reference image used in all subtractions.

To fit the offset in each band, specify ``fit_offset=True``::

    >>> res = sncosmo.fit_lc(data, model, ['mabs', 'x1', 'c', 't0', 'z'],
    ...                      bounds={'z':(0.3, 0.7)}, fit_offset=True)

The result is that four additional parameters (one for each band) have
been added to the fit. For example::

    >>> res.params  # best-fit parameters
    {'c': 0.09009158346906926,
     'mabs': -19.733868320550926,
     'offset_sdssg': -0.029615419923067394,
     'offset_sdssi': -0.0317299805728631,
     'offset_sdssr': 0.10459915813653357,
     'offset_sdssz': 0.8082400931986695,
     't0': 55100.40166018243,
     'x1': 0.40052154978406385,
     'z': 0.5399703676493539}

The uncertainties on the offset parameters are also reported in
``res.errors``, ``res.covariance``, and ``res.matrix``. The values are
flux in some units that correspond to a given magnitude system and
zeropoint. By default, this is the AB system and a zeropoint
of 25. These values can be changed using the keywords ``offset_zp``
and ``offset_zpsys``. For example::

    >>> res = sncosmo.fit_lc(data, model, ['mabs', 'x1', 'c', 't0', 'z'],
    ...                      bounds={'z':(0.3, 0.7)}, fit_offset=True,
    ...                      offset_zp=27.5)

would result in best-fit values of ``offset_[band]`` that are 10 times
larger (since the zeropoint is now 2.5 times larger). They would still
correspond to the same *physical* flux values. In other words,
``offset_zp`` and ``offset_zpsys`` essentially specify the "units"
that the offset flux values are returned in.

Note that more function calls were required because we had nine
parameters to fit instead of only five::

    >>> res.ncalls
    376

Finally, an additional attribute had been added to ``res``::

    >>> res.offsets
    {'sdssg': -0.029615419923067394,
     'sdssi': -0.0317299805728631,
     'sdssr': 0.10459915813653357,
     'sdssz': 0.8082400931986695}

These values are the same as the best-fit offset values in
``res.params`` but the dictionary keys are the names of the
bandpasses. This is useful when plotting the best-fit model and data
together.

To plot the model including these offsets::

    >>> model.set(**res.params)
    >>> sncosmo.plot_lc(data, model, offsets=res.offsets)
