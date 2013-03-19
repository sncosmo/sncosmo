# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for GridData class."""

import numpy as np

from ..griddata import GridData2d

def test_shapes():

    a = np.arange(10.)
    b = np.arange(10.)
    y = np.ones((10, 10))
    
    d = GridData2d(a, b, y)
    assert d(1, 1).shape == ()
    assert d(a, 1).shape == (len(a), 1)
    assert d(1, b).shape == (len(b),)
    assert d([1], b).shape == (1, len(b))
    assert d(a, b).shape == (len(a), len(b))
    assert d().shape == (len(a), len(b))
    assert d(None, b).shape == (len(a), len(b))
    assert d(a, None).shape == (len(a), len(b))
