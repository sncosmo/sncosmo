"""
================================
Using a custom fitter or sampler
================================

How to use your own minimizer or MCMC sampler for fitting light curves.

SNCosmo has three functions for model parameter estimation based on
photometric data: `sncosmo.fit_lc`, `sncosmo.mcmc_lc` and
`sncosmo.nest_lc`. These are wrappers around external minimizers or
samplers (respectively: iminuit, emcee and nestle). However, one may
wish to experiment with a custom fitting or sampling method.

Here, we give a minimal example of using the L-BFGS-B minimizer from scipy.
"""

from __future__ import print_function

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import sncosmo

model = sncosmo.Model(source='salt2')
data = sncosmo.load_example_data()

# Define an objective function that we will pass to the minimizer.
# The function arguments must comply with the expectations of the specfic
# minimizer you are using.
def objective(parameters):
    model.parameters[:] = parameters  # set model parameters

    # evaluate model fluxes at times/bandpasses of data
    model_flux = model.bandflux(data['band'], data['time'],
                                zp=data['zp'], zpsys=data['zpsys'])

    # calculate and return chi^2
    return np.sum(((data['flux'] - model_flux) / data['fluxerr'])**2)

# starting parameter values in same order as `model.param_names`:
start_parameters = [0.4, 55098., 1e-5, 0., 0.]  # z, t0, x0, x1, c

# parameter bounds in same order as `model.param_names`:
bounds = [(0.3, 0.7), (55080., 55120.), (None, None), (None, None),
          (None, None)]

parameters, val, info = fmin_l_bfgs_b(objective, start_parameters,
                                      bounds=bounds, approx_grad=True)

print(parameters)


#####################################################################
# The built-in parameter estimation functions in sncosmo take care of
# setting up the likelihood function in the way that the underlying
# fitter or sampler expects. Additionally, they set guesses and bounds
# and package results up in a way that is as consistent as
# possible. For users wishing use a custom minimizer or sampler, it
# can be instructive to look at the source code for these functions.
