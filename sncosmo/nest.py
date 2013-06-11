# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Simple implementation of nested sampling routine."""

import math
import time
from sys import stdout

import numpy as np

def randsphere(n):
    """Draw a random point within a n-dimensional unit sphere"""

    z = np.random.randn(n)
    return z * np.random.rand()**(1./n) / np.sqrt(np.sum(z**2))

def ellipsoid(X, expand=1.):
    """
    Calculate ellipsoid containing all samples X.

    Parameters
    ----------
    X : (nobj, ndim) ndarray

    Returns
    -------
    vs : (ndim, ndim) ndarray
        Scaled eigenvectors (in columns): vs[:,i] is the i-th eigenvector.
    mean : (ndim,) ndarray
        Simple average of all samples.
    """

    X_avg = np.mean(X, axis=0)
    Xp = X - X_avg
    c = np.cov(Xp, rowvar=0)
    cinv = np.linalg.inv(c)
    w, v = np.linalg.eig(c)
    vs = np.dot(v, np.diag(np.sqrt(w)))  # scaled eigenvectors

    # Calculate 'k' factor
    k = np.empty(len(X), dtype=np.float)

    #for i in range(len(k)):
    #    k[i] = np.dot(np.dot(Xp[i,:], cinv), Xp[i,:])
    
    # equivalent to above:
    tmp = np.tensordot(Xp, cinv, axes=1)
    for i in range(len(k)):
        k[i] = np.dot(tmp[i,:], Xp[i,:])

    k = np.max(k)

    return np.sqrt(k) * expand * vs, X_avg

def sample_ellipsoid(vs, mean, nsamples=1):
    """Chose sample(s) randomly distributed within an ellipsoid."""

    ndim = len(mean)
    if nsamples == 1:
        return np.dot(vs, randsphere(ndim)) + mean

    x = np.empty((nsamples, ndim), dtype=np.float)
    for i in range(nsamples):
        x[i, :] = np.dot(vs, randsphere(ndim)) + mean
    return x

def nest(loglikelihood, prior, npar, nobj=100, maxiter=10000,
         return_samples=False, verbose=False, verbose_name=''):
    """Simple nested sampling algorithm using a single ellipsoid.

    Parameters
    ----------
    loglikelihood : func
        Function returning log(likelihood) given parameters as a numpy array
    prior : func
        Function taking parameters in a unit cube (numpy array).
    npar : int
        Number of parameters.
    nobj : int
        Number of random samples. Larger numbers result in a more finely
        sampled posterior (more accurate evidence), but also a larger
        number of iterations required to converge. Default is 100.
    maxiter : int, optional
        Maximum number of iterations. Iteration may stop earlier if
        termination condition is reached. Default is 10000. The total number
        of likelihood evaluations will be ``nexplore * niter``. 
    nobj : int
        Number of random samples. Larger numbers result in a more finely
        sampled posterior (more accurate evidence), but also a larger
        number of iterations required to converge. Default is 100.
    nexplore : int, optional
        Number of iterations in each exploration step. Default is 20.
    return_samples : bool, optional

    verbose : bool, optional
        Print a single line of running total iterations

    Returns
    -------
    results : dict
        Containing keys `'niter'` (int, number of iterations),
        `'parnames'`, `'parvals'`, `'parstds'`,
        `'logz'`, `'logzstd'`, `'h'`, and optionally: `'samples'`,
        `'sampleswt'`
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

    # Initialize objects and calculate likelihoods
    objects_u = np.random.random((nobj, npar)) #position in unit cube
    objects_v = np.empty((nobj, npar), dtype=np.float) #position in unit cube
    objects_logl = np.empty(nobj, dtype=np.float)  # log likelihood
    for i in range(nobj):
        objects_v[i,:] = prior(objects_u[i,:])
        objects_logl[i] = loglikelihood(objects_v[i,:])

    # Initialize values for nested sampling loop.
    samples_parvals = [] # stored objects for posterior results
    samples_logwt = []
    loglstar = None  # ln(Likelihood constraint)
    h = 0.  # Information, initially 0.
    logz = -1.e300  # ln(Evidence Z, initially 0)
    # ln(width in prior mass), outermost width is 1 - e^(-1/n)
    logwidth = math.log(1. - math.exp(-1./nobj))
    loglcalls = nobj #number of calls we already made

    # Nested sampling loop.
    ndecl = 0
    logwt_old = None
    time0 = time.time()
    for it in range(maxiter):
        if verbose:
            if logz > -1.e6:
                print "\r{} iter={:6d} logz={:8f}".format(verbose_name, it,
                                                          logz),
            else:
                print "\r{} iter={:6d} logz=".format(verbose_name, it),
            stdout.flush()

        # worst object in collection and its weight (= width * likelihood)
        worst = np.argmin(objects_logl)
        logwt = logwidth + objects_logl[worst]

        # update evidence Z and information h.
        logz_new = np.logaddexp(logz, logwt)
        h = (math.exp(logwt - logz_new) * objects_logl[worst] +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new

        # Add worst object to samples.
        samples_parvals.append(np.array(objects_v[worst]))
        samples_logwt.append(logwt)

        # The new likelihood constraint is that of the worst object.
        loglstar = objects_logl[worst]

        # calculate the ellipsoid in parameter space that contains all the
        # samples (including the worst one).
        vs, mean = ellipsoid(objects_u, expand=1.06)

        # choose a point from within the ellipse until it has likelihood
        # better than loglstar
        while True:
            u = sample_ellipsoid(vs, mean)
            if np.any(u < 0.) or np.any(u > 1.):
                continue
            v = prior(u)
            logl = loglikelihood(v)
            loglcalls += 1

            # Accept if and only if within likelihood constraint.
            if logl > loglstar:
                objects_u[worst] = u
                objects_v[worst] = v
                objects_logl[worst] = logl
                break

        # Shrink interval
        logwidth -= 1./nobj

        # stop when the logwt has been declining for more than 10 or niter/4
        # consecutive iterations.
        if logwt < logwt_old:
            ndecl += 1
        else:
            ndecl = 0
        if ndecl > 10 and ndecl > it / 6:
            break
        logwt_old = logwt

    if verbose:
        time1 = time.time()
        print 'calls={:d} time={:6.2f}s'.format(loglcalls, time1 - time0)

    # Add remaining objects.
    # After N samples have been taken out, the remaining width is e^(-N/nobj)
    # The remaining width for each object is e^(-N/nobj) / nobj
    # The log of this for each object is:
    # log(e^(-N/nobj) / nobj) = -N/nobj - log(nobj)
    logwidth = -len(samples_parvals) / nobj - math.log(nobj)
    for i in range(nobj):
        logwt = logwidth + objects_logl[i]
        logz_new = np.logaddexp(logz, logwt)
        h = (math.exp(logwt - logz_new) * objects_logl[i] +
             math.exp(logz - logz_new) * (h + logz) -
             logz_new)
        logz = logz_new
        samples_parvals.append(np.array(objects_v[i]))
        samples_logwt.append(logwt)

    # process samples and return
    niter = it + 1
    nsamples = len(samples_parvals)
    samples_parvals = np.array(samples_parvals)  # (nsamp, npar)
    samples_logwt = np.array(samples_logwt)
    w = np.exp(samples_logwt - logz)  # Proportional weights.
    parvals = np.average(samples_parvals, axis=0, weights=w)
    parstds = np.sqrt(np.sum(w[:, np.newaxis] * samples_parvals**2, axis=0) -
                      parvals**2)
    logzstd = math.sqrt(h/nobj)

    result =  {'niter': niter,
               'nsamples': nsamples,
               'parvals': parvals,
               'parerrs': parstds,
               'logz': logz,
               'logzerr': logzstd,
               'h': h}
    if return_samples:
        result['samples_parvals'] = samples_parvals
        result['samples_wt'] = w
    return result
