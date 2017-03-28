import os
import pickle

from astropy.extern import six
import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import RectBivariateSpline

import sncosmo
from sncosmo.salt2utils import BicubicInterpolator, SALT2ColorLaw


# On Python 2 highest protocol is 2.
# Protocols 0 and 1 don't work on the classes here!
TEST_PICKLE_PROTOCOLS = (2,) if six.PY2 else (2, 3, 4)


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

    assert f(0., [1., 2.]).shape == f2(0., [1., 2.]).shape
    assert f([1., 2.], 0.).shape == f2([1., 2.], 0.).shape
    assert f(0., 0.).shape == f2(0., 0.).shape


def test_bicubic_interpolator_pickle():
    x = np.arange(5)
    y = np.arange(10)
    z = np.ones((len(x), len(y)))
    f = BicubicInterpolator(x, y, z)

    for protocol in TEST_PICKLE_PROTOCOLS:
        f2 = pickle.loads(pickle.dumps(f, protocol=protocol))
        assert f2(4., 5.5) == f(4., 5.5)


def test_salt2colorlaw_vs_python():
    """Compare SALT2ColorLaw vs python implementation"""

    B_WAVELENGTH = 4302.57
    V_WAVELENGTH = 5428.55
    colorlaw_coeffs = [-0.504294, 0.787691, -0.461715, 0.0815619]
    colorlaw_range = (2800., 7000.)

    # old python implementation
    def colorlaw_python(wave):
        v_minus_b = V_WAVELENGTH - B_WAVELENGTH

        l = (wave - B_WAVELENGTH) / v_minus_b
        l_lo = (colorlaw_range[0] - B_WAVELENGTH) / v_minus_b
        l_hi = (colorlaw_range[1] - B_WAVELENGTH) / v_minus_b

        alpha = 1. - sum(colorlaw_coeffs)
        coeffs = [0., alpha]
        coeffs.extend(colorlaw_coeffs)
        coeffs = np.array(coeffs)
        prime_coeffs = (np.arange(len(coeffs)) * coeffs)[1:]

        extinction = np.empty_like(wave)

        # Blue side
        idx_lo = l < l_lo
        p_lo = np.polyval(np.flipud(coeffs), l_lo)
        pprime_lo = np.polyval(np.flipud(prime_coeffs), l_lo)
        extinction[idx_lo] = p_lo + pprime_lo * (l[idx_lo] - l_lo)

        # Red side
        idx_hi = l > l_hi
        p_hi = np.polyval(np.flipud(coeffs), l_hi)
        pprime_hi = np.polyval(np.flipud(prime_coeffs), l_hi)
        extinction[idx_hi] = p_hi + pprime_hi * (l[idx_hi] - l_hi)

        # In between
        idx_between = np.invert(idx_lo | idx_hi)
        extinction[idx_between] = np.polyval(np.flipud(coeffs), l[idx_between])

        return -extinction

    colorlaw = SALT2ColorLaw(colorlaw_range, colorlaw_coeffs)

    wave = np.linspace(2000., 9200., 201)
    assert np.all(colorlaw(wave) == colorlaw_python(wave))


def test_salt2colorlaw_pickle():

    colorlaw_coeffs = [-0.504294, 0.787691, -0.461715, 0.0815619]
    colorlaw_range = (2800., 7000.)
    colorlaw = SALT2ColorLaw(colorlaw_range, colorlaw_coeffs)

    for protocol in TEST_PICKLE_PROTOCOLS:
        colorlaw2 = pickle.loads(pickle.dumps(colorlaw, protocol=protocol))
        wave = np.linspace(2000., 9200., 201)
        assert np.all(colorlaw(wave) == colorlaw2(wave))
