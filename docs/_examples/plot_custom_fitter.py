"""
================================
Using a custom fitter or sampler
================================

How to use your own minimizer or MCMC sampler for fitting light curves.

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
like this:
"""

import sncosmo

model = sncosmo.Model(source='salt2')

def loglikelihood(parameters):
    model.parameters[:] = parameters  # set model parameters
    return -0.5 * sncosmo.chisq(data, model)

##########################################################################
# What does the `sncosmo.chisq` function do? The definition is basically::
#
#    mflux = model.bandflux(data['band'], data['time'],
#                           zp=data['zp'], zpsys=data['zpsys'])
#    return np.sum(((data['flux'] - mflux) / data['fluxerr'])**2)
#
# In other words, "use the model to predict the flux in the given
# bandpasses and times (scaled with the appropriate zeropoint), then use
# those values to calculate the chisq." So really `Model.bandflux` is
# the key method that makes this all possible.
#
# You might notice that our ``loglikelihood`` function above will vary *all*
# the parameters in the model, which might not be what you want. To only vary
# select parameters, you could do something like this:

# Indicies of the model parameters that should be varied
# idx = np.array([model.param_names.index(name)
#                 for name in param_names_to_vary])

def loglikelihood(parameters):
    model.parameters[idx] = parameters
    return -0.5 * sncosmo.chisq(data, model)

#####################################################################
# The built-in wrapper functions in sncosmo don't do anything much
# more complex than this. They take care of setting up the likelihood
# function in the way that the underlying fitter or sampler
# expects. They set guesses and bounds and package results up in a way
# that is as consistent as possible across the three functions.
