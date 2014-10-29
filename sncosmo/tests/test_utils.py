# Licensed under a 3-clause BSD style license - see LICENSES

import numpy as np
from numpy.testing import assert_allclose, assert_approx_equal

from sncosmo import utils


def test_weightedcov():

    x = np.random.random((10, 3))
    w = np.random.random((10,))

    mean, cov = utils.weightedcov(x, w)

    # check individual elements
    xd = x - np.average(x, weights=w, axis=0)
    prefactor = w.sum() / (w.sum()**2 - (w**2).sum())
    ans00 = prefactor * np.sum(w * xd[:, 0] * xd[:, 0])
    assert_approx_equal(cov[0, 0], ans00)
    ans01 = prefactor * np.sum(w * xd[:, 0] * xd[:, 1])
    assert_approx_equal(cov[0, 1], ans01)

    # If weights are all equal, covariance should come out to simple case
    w = np.repeat(0.2, 10)
    mean, cov = utils.weightedcov(x, w)
    assert_allclose(cov, np.cov(x, rowvar=0))
    assert_allclose(mean, np.average(x, axis=0))
