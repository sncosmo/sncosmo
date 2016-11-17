# Licensed under a 3-clause BSD style license - see LICENSES

from numpy.testing import assert_allclose

from sncosmo import Bandpass


def test_bandpass_zeros():
    assert_allclose(Bandpass([1., 2., 3., 4., 5.], [0., 0., 1., 1., 1.]).wave,
                    [2., 3., 4., 5.])
    assert_allclose(Bandpass([1., 2., 3., 4., 5.], [0., 1., 1., 1., 1.]).wave,
                    [1., 2., 3., 4., 5.])
    assert_allclose(Bandpass([1., 2., 3., 4., 5.], [1., 1., 1., 1., 1.]).wave,
                    [1., 2., 3., 4., 5.])

    assert_allclose(Bandpass([1., 2., 3., 4., 5.], [0., 0., 1., 1., 0.]).wave,
                    [2., 3., 4., 5.])
    assert_allclose(Bandpass([1., 2., 3., 4., 5.], [0., 0., 1., 0., 0.]).wave,
                    [2., 3., 4.])
