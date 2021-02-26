# Licensed under a 3-clause BSD style license - see LICENSES
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

import sncosmo
from sncosmo import Bandpass
from sncosmo.tests.test_salt2source import read_header


def test_bandpass_access():
    b = Bandpass([4000., 4200., 4400.], [0.5, 1.0, 0.5])
    assert_allclose(b.wave, [4000., 4200., 4400.])
    assert_allclose(b.trans, [0.5,  1.,  0.5])


def test_bandpass_interpolation():
    b = Bandpass([4000., 4200., 4400.], [0.5, 1.0, 0.5])
    assert_allclose(b([4100., 4300.]), [0.75,  0.75])


def test_bandpass_effective_wavelength():
    b = Bandpass([4000., 4200., 4400.], [0.5, 1.0, 0.5])
    assert b.wave_eff == 4200.0


def test_bandpass_zeros():
    """Test that removing outlying zeros works as expected."""
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


def test_trimmed():
    band = Bandpass([4000., 4100., 4200., 4300., 4400., 4500.],
                    [0.001, 0.002,   0.5,   0.6, 0.003, 0.001],
                    trim_level=0.01)

    assert np.all(band.wave == np.array([4100.,  4200.,  4300.,  4400.]))
    assert_allclose(band.trans,  np.array([0.002,  0.5,  0.6,  0.003]))


# issue 100
def test_bandpass_type():
    """Check that bandpass wavelength type is always float64,
    and that color laws work with them."""

    dust = sncosmo.CCM89Dust()

    for dt in [np.int32, np.int64, np.float32]:
        wave = np.arange(4000., 5000., 20., dtype=dt)
        trans = np.ones_like(wave)
        band = sncosmo.Bandpass(wave, trans)

        assert band.wave.dtype == np.float64

        # Ensure that it works with cython-based propagation effect.
        # (flux, the second argument, should always be doubles)
        dust.propagate(band.wave, np.ones_like(wave, dtype=np.float64))


# issue 111
def test_bandpass_bessell():
    """Check that Bessell bandpass definitions are scaled by inverse
    wavelength."""

    band = sncosmo.get_bandpass('bessellb')
    trans = band.trans[[4, 9, 14]]  # transmission at 4000, 4500, 5000

    # copied from file
    orig_wave = np.array([4000., 4500., 5000.])
    orig_trans = np.array([0.920, 0.853, 0.325])

    scaled_trans = orig_trans / orig_wave

    # scaled_trans should be proportional to trans
    factor = scaled_trans[0] / trans[0]
    assert_allclose(scaled_trans, factor * trans)


def test_aggregate_bandpass_name():
    b = sncosmo.AggregateBandpass([([1000., 2000.], [1., 1.])])
    assert repr(b).startswith("<AggregateBandpass")


@pytest.mark.might_download
def test_megacampsf_bandpass():
    """Test megacampsf position-dependent bandpasses against snfit"""
    dirname = os.path.join(os.path.dirname(__file__), "data")

    for letter in ('g', 'z'):
        for i in (0, 1):
            fname = os.path.join(
                dirname, 'snfit_filter_{:s}_{:d}.dat'.format(letter, i))

            with open(fname, 'r') as f:
                meta = read_header(f)
                wave, trans_ref = np.loadtxt(f, unpack=True)

            # sncosmo version of bandpass:
            band = sncosmo.get_bandpass('megacampsf::'+letter, meta['radius'])
            trans = band(wave)
            for i in range(len(trans)):
                print(trans_ref[i], trans[i])
            assert_allclose(trans, trans_ref, rtol=1e-5)
