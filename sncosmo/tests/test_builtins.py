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
