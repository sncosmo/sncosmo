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
access. You can see what attributes it has::

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
