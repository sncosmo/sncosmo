# Licensed under a 3-clause BSD style license - see LICENSES
from __future__ import print_function

from os.path import dirname, join

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


def test_read_salt2():
    fname = join(dirname(__file__), "data", "salt2_example.dat")
    data = sncosmo.read_lc(fname, format="salt2")

    # Test a few columns
    assert_allclose(data["Date"], [52816.54, 52824.59, 52795.59, 52796.59])
    assert_allclose(data["ZP"], [27.091335, 27.091335, 25.913054, 25.913054])
    assert np.all(data["Filter"] == np.array(["MEGACAM::g", "MEGACAM::g",
                                              "MEGACAM::i", "MEGACAM::i"]))
    assert np.all(data["MagSys"] == "VEGA")

    # Test a bit of metadata
    assert_allclose(data.meta["Z_HELIO"], 0.285)
    assert_allclose(data.meta["RA"], 333.690959)
    assert data.meta["z_source"] == "H"
