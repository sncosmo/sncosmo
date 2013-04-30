*******************
Light Curve Fitting
*******************

Fit model parameters to data
----------------------------

    >>> from sncosmo.fitting import fit_model
    >>> data = {'time': ...
    
    >>> model = sncosmo.get_model('salt2')
    >>> model.set(z=0.2)
    >>> fit_model(model, data, ['m', 'c', 'x1', 't0'])
    >>> model.params
