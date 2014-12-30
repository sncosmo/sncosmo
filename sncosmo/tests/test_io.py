# Licensed under a 3-clause BSD style license - see LICENSES
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from astropy.extern import six

import sncosmo


def test_read_griddata_ascii():

    # Write a temporary test file.
    f = six.StringIO()
    f.write("0. 0. 0.\n"
            "0. 1. 0.\n"
            "0. 2. 0.\n"
            "1. 0. 0.\n"
            "1. 1. 0.\n"
            "1. 2. 0.\n")
    f.seek(0)

    x0, x1, y = sncosmo.read_griddata_ascii(f)

    assert_allclose(x0, np.array([0., 1.]))
    assert_allclose(x1, np.array([0., 1., 2.]))


def test_write_griddata_ascii():

    x0 = np.array([0., 1.])
    x1 = np.array([0., 1., 2.])
    y = np.zeros((2, 3))

    f = six.StringIO()
    sncosmo.write_griddata_ascii(x0, x1, y, f)

    # Read it back
    f.seek(0)
    x0_in, x1_in, y_in = sncosmo.read_griddata_ascii(f)
    assert_allclose(x0_in, x0)
    assert_allclose(x1_in, x1)
    assert_allclose(y_in, y)

def test_griddata_fits():
    """Round tripping with write_griddata_fits() and read_griddata_fits()"""

    x0 = np.array([0., 1.])
    x1 = np.array([0., 1., 2.])
    y = np.zeros((2, 3))

    f = six.StringIO()
    sncosmo.write_griddata_fits(x0, x1, y, f)

    # Read it back
    f.seek(0)
    x0_in, x1_in, y_in = sncosmo.read_griddata_fits(f)
    assert_allclose(x0_in, x0)
    assert_allclose(x1_in, x1)
    assert_allclose(y_in, y)
