# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for extinction curve."""

import numpy as np
from sncosmo.extinction import extinction_ccm

def test_extinction_ccm_shapes():

    # Test single value
    extinction_ccm(1.e4, a_v=1.)

    # multiple values
    assert extinction_ccm([1.e4], a_v=1.).shape == (1,)
    assert extinction_ccm([1.e4, 2.e4], a_v=1.).shape == (2,)


def test_extinction_ccm_values():

    r_v = 3.1

    # U, B, V, R, I, J, H, K band effective wavelengths from CCM '89 table 3
    x_inv_microns = np.array([2.78, 2.27, 1.82, 1.43, 1.11, 0.80, 0.63, 0.46])

    # A(lambda)/A(V) for R_V = 3.1 from Table 3 of CCM '89
    ccm_ratio_true = np.array([1.569, 1.337, 1.000, 0.751, 0.479, 0.282,
                               0.190, 0.114])

    wavelengths = 1.e4 / x_inv_microns  # wavelengths in Angstroms
    ccm_ratio = extinction_ccm(wavelengths, r_v=3.1, a_v=1.,
                               optical_coeffs='ccm')

    # TODO:
    # So far, these are close but not exact.
    # I get: [ 1.56880904  1.32257836  1. 0.75125994  0.4780346   0.28206957
    #          0.19200814  0.11572348]

    # At the sigfigs of Table 3, the differences are:
    # [ None, 0.014, None, None, 0.001, None, 0.002, 0.002 ]
    # with B band being the most significant difference.

    # Could be a floating point issue, due to the polynomials? Maybe in the 
    # original Cardelli paper? Should compare a and b to the values in the
    # table.
