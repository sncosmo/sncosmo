import os

import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import RectBivariateSpline

import sncosmo
from sncosmo.interpolation import BicubicInterpolator


def test_bicubic_interpolator_vs_snfit():
    datadir = os.path.join(os.path.dirname(__file__), "data")

    # created by running generate script in `misc` directory
    fname_input = os.path.join(datadir, "interpolation_test_input.dat")
    fname_evalx = os.path.join(datadir, "interpolation_test_evalx.dat")
    fname_evaly = os.path.join(datadir, "interpolation_test_evaly.dat")

    # result file was created by running snfit software Grid2DFunction
    fname_result = os.path.join(datadir, "interpolation_test_result.dat")

    # load arrays
    x, y, z = sncosmo.read_griddata_ascii(fname_input)
    xp = np.loadtxt(fname_evalx)
    yp = np.loadtxt(fname_evaly)
    result = np.loadtxt(fname_result)

    f = BicubicInterpolator(x, y, z)
    assert_allclose(f(xp, yp), result, rtol=1e-5)


def test_bicubic_interpolator_shapes():
    """Ensure that input shapes are handled like RectBivariateSpline"""

    x = np.array([1., 2., 3., 4., 5.])
    z = np.ones((len(x), len(x)))

    f = BicubicInterpolator(x, x, z)
    f2 = RectBivariateSpline(x, x, z)

    assert f(0., [1., 2.]).shape == f2(0., [1.,2.]).shape
    assert f([1., 2.], 0.).shape == f2([1., 2.], 0.).shape
    assert f(0., 0.).shape == f2(0., 0.).shape
