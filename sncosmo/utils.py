import math

import numpy as np
from scipy import integrate, optimize


def format_value(value, error=None, latex=False):
    """Return a string representing value and uncertainty.

    If latex=True, use '\pm' and '\times'.
    """

    if latex:
        pm = '\pm'
        suffix_templ = ' \\times 10^{{{0:d}}}'
    else:
        pm = '+/-'
        suffix_templ = ' x 10^{0:d}'

    # First significant digit
    absval = abs(value)
    if absval == 0.:
        first = 0
    else:
        first = int(math.floor(math.log10(absval)))

    if error is None or error == 0.:
        last = first - 6  # Pretend there are 7 significant figures.
    else:
        last = int(math.floor(math.log10(error)))  # last significant digit

    # use exponential notation if
    # value > 1000 and error > 1000 or value < 0.01
    if (first > 2 and last > 2) or first < -2:
        value /= 10**first
        if error is not None:
            error /= 10**first
        p = max(0, first - last + 1)
        suffix = suffix_templ.format(first)
    else:
        p = max(0, -last + 1)
        suffix = ''

    if error is None:
        prefix = ('{0:.' + str(p) + 'f}').format(value)
    else:
        prefix = (('{0:.' + str(p) + 'f} {1:s} {2:.' + str(p) + 'f}')
                  .format(value, pm, error))
        if suffix != '':
            prefix = '({0})'.format(prefix)

    return prefix + suffix


class Result(dict):
    """Represents an optimization result.

    Notes
    -----
    This is a cut and paste from scipy, normally imported with `from
    scipy.optimize import Result`. However, it isn't available in
    scipy 0.9 (or possibly 0.10), so it is included here.
    Since this class is essentially a subclass of dict with attribute
    accessors, one can see which attributes are available using the
    `keys()` method.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"


def _integral_diff(x, pdf, a, q):
    """Return difference between q and the integral of the function `pdf`
    between a and x. This is used for solving for the ppf."""
    return integrate.quad(pdf, a, x)[0] - q


def ppf(pdf, x, a, b):
    """Percent-point function (inverse cdf), given the probability
    distribution function pdf and limits a, b.

    Parameters
    ----------
    pdf : callable
        Probability distribution function
    x : array_like
        Points at which to evaluate the ppf
    a, b : float
        Limits (can be -np.inf, np.inf, assuming pdf has finite integral).
    """

    FACTOR = 10.

    if not b > a:
        raise ValueError('b must be greater than a')

    # integral of pdf between a and b
    tot = integrate.quad(pdf, a, b)[0]

    # initialize result array
    x = np.asarray(x)
    shape = x.shape
    x = np.ravel(x)
    result = np.zeros(len(x))

    for i in range(len(x)):
        cumsum = x[i] * tot  # target cumulative sum
        left = a
        right = b

        # Need finite limits for the solver.
        # For inifinite upper or lower limits, find finite limits such that
        # cdf(left) < cumsum < cdf(right).
        if left == -np.inf:
            left = -FACTOR
            while integrate.quad(pdf, a, left)[0] > cumsum:
                right = left
                left *= FACTOR
        if right == np.inf:
            right = FACTOR
            while integrate.quad(pdf, a, right)[0] < cumsum:
                left = right
                right *= FACTOR

        result[i] = optimize.brentq(_integral_diff, left, right,
                                    args=(pdf, a, cumsum))

    return result.reshape(shape)


class Interp1D(object):
    def __init__(self, xmin, xmax, y):
        self._xmin = xmin
        self._xmax = xmax
        self._n = len(y)
        self._xstep = (xmax - xmin) / (self._n - 1)
        self._y = y

    def __call__(self, x):
        """works only in range [xmin, xmax)"""
        nsteps = (x - self._xmin) / self._xstep
        i = int(nsteps)
        w = nsteps - i
        return (1.-w) * self._y[i] + w * self._y[i+1]


def weightedcov(x, w):
    """Estimate a covariance matrix, given data with weights.

    Implements formula described here:
    https://en.wikipedia.org/wiki/Sample_mean_and_sample_covariance
    (see "weighted samples" section)

    Parameters
    ----------
    x : `~numpy.ndarray`
        2-D array containing data samples. Shape is (M, N) where N is the
        number of variables and M is the number of samples or observations.
    w : `~numpy.ndarray`
        1-D array of sample weights. Shape is (M,).

    Returns
    -------
    mean : `~numpy.ndarray`
        Weighted mean of samples.
    cov : `~numpy.ndarray`
        Weighted covariance matrix.
    """

    xmean = np.average(x, weights=w, axis=0)
    xd = x - xmean
    wsum = np.sum(w)
    w2sum = np.sum(w**2)

    cov = wsum / (wsum**2 - w2sum) * np.einsum('i,ij,ik', w, xd, xd)

    return xmean, cov
