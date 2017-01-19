# Licensed under a 3-clause BSD style license - see LICENSES

from tempfile import mkdtemp
import os

import numpy as np
from numpy.testing import assert_allclose, assert_approx_equal
from scipy.stats import norm
import pytest

from sncosmo import utils


def test_result():
    res = utils.Result(a=1, b=2)
    assert res.a == 1
    assert res.b == 2

    # test deprecating result attributes
    res.__dict__['deprecated']['c'] = (2, "Use b instead")

    # can't test warnings currently because bundled version of pytest in
    # astropy is too old to support pytest.warns (need 2.8)
    assert res.c == 2


def test_format_value():
    assert utils.format_value(1.234567) == '1.2345670'
    assert utils.format_value(0.001234567) == '1.2345670 x 10^-3'
    assert utils.format_value(1234567, error=1) == '1234567.0 +/- 1.0'
    assert (utils.format_value(0.001234567, latex=True) ==
            '1.2345670 \\times 10^{-3}')


def test_ppf():
    """Test the ppf function."""

    # Flat prior between 0 and 10
    def prior(x):
        return 1.

    x = np.array([0.1, 0.2, 0.9, 0.9999])
    y = utils.ppf(prior, x, 0., 10.)
    assert_allclose(y, [1., 2., 9., 9.999])

    # test a normal distribution
    priordist = norm(0., 1.)
    x = np.linspace(0.05, 0.95, 5.)
    y = utils.ppf(priordist.pdf, x, -np.inf, np.inf)
    assert_allclose(y, priordist.ppf(x), atol=1.e-10)


def test_alias_map():
    mapping = utils.alias_map(['A', 'B_', 'foo'],
                              {'a': set(['a', 'a_']), 'b': set(['b', 'b_'])})
    assert mapping == {'a': 'A', 'b': 'B_'}


def test_data_mirror_rootdir():
    dirname = mkdtemp()

    # rootdir is a string
    mirror = utils.DataMirror(dirname, "url_goes_here")
    assert mirror.rootdir() == dirname

    # rootdir is a callable
    mirror = utils.DataMirror(lambda: dirname, "url_goes_here")
    assert mirror.rootdir() == dirname

    os.rmdir(dirname)
