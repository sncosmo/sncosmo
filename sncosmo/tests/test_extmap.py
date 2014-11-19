# Licensed under a 3-clause BSD style license - see LICENSES

from os.path import dirname, join

import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord

import sncosmo

def test_get_ebv_from_map():
    """Try get_ebv_from_map."""

    mapdir = join(dirname(__file__), "data", "dust_subsection")

    # coordinate that is in the dust subsection map
    coords = (204., -30.)
    coords2 = SkyCoord(204.0, -30.0, frame='icrs', unit='deg')

    # value from http://irsa.ipac.caltech.edu/applications/DUST/
    # with input "204.0 -30.0 J2000"
    true_ebv = 0.0477

    # Use interpolate=False to match IRSA value
    ebv = sncosmo.get_ebv_from_map(coords, mapdir=mapdir, interpolate=False)
    assert_allclose(ebv, true_ebv, rtol=0.01)
    ebv2 = sncosmo.get_ebv_from_map(coords2, mapdir=mapdir, interpolate=False)
    assert_allclose(ebv2, true_ebv, rtol=0.01)
