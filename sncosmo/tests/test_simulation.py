# Licensed under a 3-clause BSD style license - see LICENSES
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from astropy.table import Table

import sncosmo
from .test_models import flatsource


def test_zdist():
    """Test that zdist function works."""

    np.random.seed(0)
    z = list(sncosmo.zdist(0., 0.25))

    # check that zdist returns the same number of SNe as
    # when this test was written (does not test whether the number is
    # correct)
    assert len(z) == 14

    # check that all values are indeed between the input limits.
    zarr = np.array(z)
    assert np.all((zarr > 0.) & (zarr < 0.25))


def test_realize_lcs():

    # here's some completely made-up data:
    obs1 = Table({'time': [10., 60., 110.],
                  'band': ['bessellb', 'bessellr', 'besselli'],
                  'gain': [1., 1., 1.],
                  'skynoise': [100., 100., 100.],
                  'zp': [30., 30., 30.],
                  'zpsys': ['ab', 'ab', 'ab']})

    # same made up data with aliased column names:
    obs2 = Table({'MJD': [10., 60., 110.],
                  'filter': ['bessellb', 'bessellr', 'besselli'],
                  'GAIN': [1., 1., 1.],
                  'skynoise': [100., 100., 100.],
                  'ZPT': [30., 30., 30.],
                  'zpmagsys': ['ab', 'ab', 'ab']})

    for obs in (obs1, obs2):

        # A model with a flat spectrum between 0 and 100 days.
        model = sncosmo.Model(source=flatsource())

        # parameters to run
        params = [{'amplitude': 1., 't0': 0., 'z': 0.},
                  {'amplitude': 1., 't0': 100., 'z': 0.},
                  {'amplitude': 1., 't0': 200., 'z': 0.}]

        # By default, realize_lcs should return all observations for all SNe
        lcs = sncosmo.realize_lcs(obs, model, params)
        assert len(lcs[0]) == 3
        assert len(lcs[1]) == 3
        assert len(lcs[2]) == 3

        # For trim_obervations=True, only certain observations will be
        # returned.
        lcs = sncosmo.realize_lcs(obs, model, params, trim_observations=True)
        assert len(lcs[0]) == 2
        assert len(lcs[1]) == 1
        assert len(lcs[2]) == 0
