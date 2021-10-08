import pytest

import sncosmo


@pytest.mark.might_download
def test_hst_bands():
    """  check that the HST and JWST bands are accessible """
    for bandname in ['f606w', 'uvf606w', 'f125w', 'f127m',
                     'f115w']:  # jwst nircam
        sncosmo.get_bandpass(bandname)


@pytest.mark.might_download
def test_jwst_miri_bands():
    for bandname in ['f1130w']:
        sncosmo.get_bandpass(bandname)


@pytest.mark.might_download
def test_ztf_bandpass():
    bp = sncosmo.get_bandpass('ztfg')


@pytest.mark.might_download
def test_roman_bandpass():
    sncosmo.get_bandpass('f062')
    sncosmo.get_bandpass('f087')
    sncosmo.get_bandpass('f106')
    sncosmo.get_bandpass('f129')
    sncosmo.get_bandpass('f158')
    sncosmo.get_bandpass('f184')
    sncosmo.get_bandpass('f213')
    sncosmo.get_bandpass('f146')
