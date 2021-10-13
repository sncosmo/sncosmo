import pytest

import sncosmo


@pytest.mark.might_download
def test_builtins_bessell():
    sncosmo.get_bandpass('bessellb')


@pytest.mark.might_download
def test_builtins_remote_aa():
    sncosmo.get_bandpass('standard::u')


@pytest.mark.might_download
def test_builtins_remote_nm():
    sncosmo.get_bandpass('kepler')


@pytest.mark.might_download
def test_builtins_remote_um():
    sncosmo.get_bandpass('f070w')


@pytest.mark.might_download
def test_builtins_remote_wfc3():
    sncosmo.get_bandpass('f098m')


@pytest.mark.might_download
def test_builtins_tophat_um():
    sncosmo.get_bandpass('f1065c')


@pytest.mark.might_download
def test_builtins_megacampsf():
    sncosmo.get_bandpass('megacampsf::u', 0.)


@pytest.mark.might_download
def test_builtins_timeseries_ascii():
    sncosmo.get_source('nugent-sn1a')


@pytest.mark.might_download
def test_builtins_timeseries_fits():
    sncosmo.get_source('hsiao')


@pytest.mark.might_download
def test_builtins_timeseries_fits_local():
    sncosmo.get_source('hsiao-subsampled')


@pytest.mark.might_download
def test_builtins_salt2model():
    sncosmo.get_source('salt2')


@pytest.mark.might_download
def test_builtins_salt3model():
    sncosmo.get_source('salt3')


@pytest.mark.might_download
def test_builtins_2011fe():
    sncosmo.get_source('snf-2011fe')


@pytest.mark.might_download
def test_builtins_mlcs2k2():
    sncosmo.get_source('mlcs2k2')


@pytest.mark.might_download
def test_builtins_snemo():
    sncosmo.get_source('snemo2')


@pytest.mark.might_download
def test_builtins_sugar():
    sncosmo.get_source('sugar')


@pytest.mark.might_download
def test_builtins_magsys_ab():
    sncosmo.get_magsystem('ab')


@pytest.mark.might_download
def test_builtins_magsys_fits():
    sncosmo.get_magsystem('vega')


@pytest.mark.might_download
def test_builtins_magsys_csp():
    sncosmo.get_magsystem('csp')


@pytest.mark.might_download
def test_builtins_magsys_ab_b12():
    sncosmo.get_magsystem('ab-b12')


@pytest.mark.might_download
def test_builtins_magsys_jla():
    sncosmo.get_magsystem('jla1')
