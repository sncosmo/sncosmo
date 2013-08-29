# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test registry functions."""

import numpy as np
import sncosmo

def test_register():
    disp = np.array([4000., 4200., 4400., 4600., 4800., 5000.])
    trans = np.array([0., 1., 1., 1., 1., 0.])
    band = sncosmo.Bandpass(disp, trans, name='tophatg')
    sncosmo.registry.register(band)

    band2 = sncosmo.get_bandpass('tophatg')

    # make sure we can get back the band we put it.
    assert band2 is band
