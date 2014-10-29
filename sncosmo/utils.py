import math

import numpy as np
from scipy import integrate, optimize


def format_value(value, error=None, latex=False):
    """Return a string representing value and uncertainty.

    If latex=True, use '\pm' and '\times'.
    """
    pm = '\pm' if latex else '+/-'
    suffix = ''

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
        if latex:
            suffix = ' \\times 10^{{{0:d}}}'.format(first)
        else:
            suffix = ' x 10^{0:d}'.format(first)
    else:
        p = max(0, -last + 1)

    if error is None:
        prefix = '{0:g}'.format(value)
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


def _cdf(pdf, x, a):
    return integrate.quad(pdf, a, x)[0]


def _ppf_to_solve(x, pdf, q, a):
    return _cdf(pdf, x, a) - q


def _ppf_single_call(pdf, q, a, b):
    left = right = None
    if a > -np.inf:
        left = a
    if b < np.inf:
        right = b

    factor = 10.

    # if lower limit is -infinity, adjust to
    # ensure that cdf(left) < q
    if left is None:
        left = -1. * factor
        while _cdf(pdf, left, a) > q:
            right = left
            left *= factor

    # if upper limit is infinity, adjust to
    # ensure that cdf(right) > q
    if right is None:
        right = factor
        while _cdf(pdf, right, a) < q:
            left = right
            right *= factor

    return optimize.brentq(_ppf_to_solve, left, right, args=(pdf, q, a))


class Interp1d(object):

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


def pdf_to_ppf(pdf, a, b, n=101):
    """Given a function representing a pdf, return a callable representing the
    inverse cdf (or ppf) of the pdf."""

    x = np.linspace(0., 1., n)
    y = np.empty(n, dtype=np.float)
    y[0] = a
    y[-1] = b
    for i in range(1, n-1):
        y[i] = _ppf_single_call(pdf, x[i], a, b)

    return Interp1d(0., 1., y)


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

    # if we only supported numpy 1.6+ we could do this:
    # cov = wsum / (wsum**2 - w2sum) * np.einsum('i,ij,ik', w, xd, xd)

    cov = np.empty((xmean.size, xmean.size), dtype=np.float64)
    for j in range(cov.shape[0]):
        for i in range(cov.shape[1]):
            cov[j, i] = np.sum(w * xd[:, i] * xd[:, j])
    cov *= wsum / (wsum**2 - w2sum)

    return xmean, cov
