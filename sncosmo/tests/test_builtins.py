import pytest

import sncosmo

"""Test all of the builtins code.

This file is designed to test that all of the code that loads builtins works,
and only tests a single example for each kind of builtin. A separate file
(test_download_builtins.py) actually makes sure that all of the builtins can be
downloaded and loaded, but that is slow and requires a lot of downloading so it
isn't included as part of the standard test suite. If you add new builtins
they should be picked up automatically by that file.

You should only add a new test to this file if you created a new loader
function.
"""


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
def test_builtins_ztf_average():
    sncosmo.get_bandpass('ztf::g')


@pytest.mark.might_download
def test_builtins_ztf_variable():
    sncosmo.get_bandpass('ztf::g', x=0, y=0, sensor_id=1)


@pytest.mark.might_download
def test_builtins_megacam_average():
    sncosmo.get_bandpass('megacam6::g')


@pytest.mark.might_download
def test_builtins_megacam_variable():
    sncosmo.get_bandpass('megacam6::g', x=1000, y=1000, sensor_id=12)


@pytest.mark.might_download
def test_builtins_hsc_average():
    sncosmo.get_bandpass('hsc::g')


@pytest.mark.might_download
def test_builtins_hsc_variable():
    sncosmo.get_bandpass('hsc::g', x=0, y=0, sensor_id=1)


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
