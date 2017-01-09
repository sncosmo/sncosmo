# Licensed under a 3-clause BSD style license - see LICENSES

import math

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from astropy import units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import remote_data
import pytest

import sncosmo


def test_abmagsystem():
    magsys = sncosmo.ABMagSystem()
    m = magsys.band_flux_to_mag(1.0, 'bessellb')
    f = magsys.band_mag_to_flux(m, 'bessellb')
    assert_almost_equal(f, 1.0)


def test_spectralmagsystem():
    """Check that SpectralMagSystem matches ABMagSystem when the spectrum is
    the same as AB."""

    # construct a spectrum with same flux as AB: 3631 x 10^{-23} erg/s/cm^2/Hz
    # Use a fine grid to reduce linear interpolation errors when integrating
    # in Spectrum.bandflux().
    wave = np.linspace(1000., 20000., 100000)  # fine grid
    flux = 3631.e-23 * np.ones_like(wave)
    unit = u.erg / u.s / u.cm**2 / u.Hz
    s = sncosmo.Spectrum(wave, flux, unit=unit)
    magsys1 = sncosmo.SpectralMagSystem(s)

    magsys2 = sncosmo.ABMagSystem()

    assert_allclose(magsys1.zpbandflux('bessellb'),
                    magsys2.zpbandflux('bessellb'))


@remote_data
def test_csp_magsystem():
    csp = sncosmo.get_magsystem('csp')

    # filter zeropoints (copied from file)
    zps = {"cspu": 13.044,
           "cspg": 15.135,
           "cspr": 14.915,
           "cspi": 14.781,
           "cspb": 14.344,
           "cspv3014": 14.540,
           "cspv3009": 14.493,
           "cspv9844": 14.450,
           "cspys": 13.921,
           "cspjs": 13.836,
           "csphs": 13.510,
           "cspk": 11.967,
           "cspyd": 13.770,
           "cspjd": 13.866,
           "csphd": 13.502}

    # The "zero point bandflux" should be the flux that corresponds to
    # magnitude zero. So, 0 = zp - 2.5 log(F)
    for band, zp in zps.items():
        assert abs(2.5 * math.log10(csp.zpbandflux(band)) - zp) < 0.015


@remote_data
def test_compositemagsystem_band_error():
    """Test that CompositeMagSystem raises an error when band is
    not in system."""

    csp = sncosmo.get_magsystem('csp')
    with pytest.raises(ValueError):
        csp.zpbandflux('desi')
