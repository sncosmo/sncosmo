"""
=====================
Fitting a light curve
=====================

This example shows how to fit the parameters of a SALT2 model to photometric
light curve data.

First, we'll load an example of some photometric data.
"""

import sncosmo

data = sncosmo.load_example_data()

print(data)

#####################################################################
# An important additional note: a table of photometric data has a
# ``band`` column and a ``zpsys`` column that use strings to identify
# the bandpass (e.g., ``'sdssg'``) and zeropoint system (``'ab'``) of
# each observation. If the bandpass and zeropoint systems in your data
# are *not* built-ins known to sncosmo, you must register the
# corresponding `~sncosmo.Bandpass` or `~sncosmo.MagSystem` to the
# right string identifier using the registry.

# create a model
model = sncosmo.Model(source='salt2')

# run the fit
result, fitted_model = sncosmo.fit_lc(
    data, model,
    ['z', 't0', 'x0', 'x1', 'c'],  # parameters of model to vary
    bounds={'z':(0.3, 0.7)})  # bounds on parameters (if any)

#####################################################################
# The first object returned is a dictionary-like object where the keys
# can be accessed as attributes in addition to the typical dictionary
# lookup like ``result['ncall']``:
print("Number of chi^2 function calls:", result.ncall)
print("Number of degrees of freedom in fit:", result.ndof)
print("chi^2 value at minimum:", result.chisq)
print("model parameters:", result.param_names)
print("best-fit values:", result.parameters)
print("The result contains the following attributes:\n", result.keys())

##################################################################
# The second object returned is a shallow copy of the input model with
# the parameters set to the best fit values. The input model is
# unchanged.

sncosmo.plot_lc(data, model=fitted_model, errors=result.errors)

#######################################################################
# Suppose we already know the redshift of the supernova we're trying to
# fit.  We want to set the model's redshift to the known value, and then
# make sure not to vary `z` in the fit.

model.set(z=0.5)  # set the model's redshift.
result, fitted_model = sncosmo.fit_lc(data, model,
                                      ['t0', 'x0', 'x1', 'c'])
sncosmo.plot_lc(data, model=fitted_model, errors=result.errors)
