# Licensed under a 3-clause BSD style license - see LICENSES

"""Tests for SUGARSource (and wrapped in Model)"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import sncosmo


def sugar_model(q0=0, q1=0, q2=0, q3=0, Av=0,
                phase=np.linspace(-5, 30, 10),
                wave=np.linspace(4000, 8000, 10)):
    """
    Give a spectral time series of SUGAR model
    for a given set of parameters.
    """
    source = sncosmo.get_source('sugar')

    mag_sugar = source._model['M0'](phase, wave)

    keys = ['ALPHA1', 'ALPHA2', 'ALPHA3', 'CCM']
    parameters = [q1, q2, q3, Av]
    for i in range(4):
        comp = source._model[keys[i]](phase, wave) * parameters[i]
        mag_sugar += comp
    # Mag AB used in the training of SUGAR.
    mag_sugar += 48.59
    wave_factor = (wave ** 2 / 299792458. * 1.e-10)
    return (q0 * 10. ** (-0.4 * mag_sugar) / wave_factor)


@pytest.mark.might_download
def test_sugarsource():
    """Test timeseries output from SUGARSource vs pregenerated timeseries
    from the original files."""

    source = sncosmo.get_source("sugar")
    model = sncosmo.Model(source)

    q1 = [-1, 0, 1, 2]
    q2 = [1, 0, -1, -2]
    q3 = [-1, 1, 0, -2]
    Av = [-0.1, 0, 0.2, 0.5]
    q0 = [10**(-0.4 * 34), 10**(-0.4 * 33),
          10**(-0.4 * 38), 10**(-0.4 * 42)]

    time = np.linspace(-5, 30, 10)
    wave = np.linspace(4000, 8000, 10)

    for i in range(len(q1)):

        fluxref = sugar_model(q0=q0[i],
                              q1=q1[i],
                              q2=q2[i],
                              q3=q3[i],
                              Av=Av[i],
                              phase=time,
                              wave=wave)

        model.set(z=0, t0=0, q0=q0[i],
                  q1=q1[i], q2=q2[i],
                  q3=q3[i], Av=Av[i])
        flux = model.flux(time, wave)
        assert_allclose(flux, fluxref, rtol=1e-13)
