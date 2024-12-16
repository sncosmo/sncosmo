import pytest

import sncosmo

from sncosmo.bandpasses import _BANDPASSES, _BANDPASS_INTERPOLATORS
from sncosmo.magsystems import _MAGSYSTEMS
from sncosmo.models import _SOURCES

"""Test downloading all of the builtins

These tests download lots of files (~1.2 GB as of Oct. 12, 2021) so they
aren't included by default with the regular tests. They can be run with
`tox -e builtins`. This will make sure that the downloads happen in a clean
environment without any caching.
"""


bandpasses = [i['name'] for i in _BANDPASSES.get_loaders_metadata()]
bandpass_interpolators = [i['name'] for i in
                          _BANDPASS_INTERPOLATORS.get_loaders_metadata()]
magsystems = [i['name'] for i in _MAGSYSTEMS.get_loaders_metadata()]
sources = [(i['name'], i['version']) for i in _SOURCES.get_loaders_metadata()]


@pytest.fixture
def all_tarfile_errors_are_fatal():
    """
    Raise tarfile.FilterError to caller (raised if tarfile.data_filter actually
    filters out any archive members)
    """
    import tarfile
    errorlevel = tarfile.TarFile.errorlevel
    try:
        tarfile.TarFile.errorlevel = 2
        yield
    finally:
        tarfile.TarFile.errorlevel = errorlevel


@pytest.mark.might_download
@pytest.mark.usefixtures("all_tarfile_errors_are_fatal")
@pytest.mark.parametrize("name", bandpasses)
def test_builtin_bandpass(name):
    sncosmo.get_bandpass(name)


@pytest.mark.might_download
@pytest.mark.usefixtures("all_tarfile_errors_are_fatal")
@pytest.mark.parametrize("name", bandpass_interpolators)
def test_builtin_bandpass_interpolator(name):
    interpolator = _BANDPASS_INTERPOLATORS.retrieve(name)
    interpolator.at(interpolator.minpos())


@pytest.mark.might_download
@pytest.mark.usefixtures("all_tarfile_errors_are_fatal")
@pytest.mark.parametrize("name,version", sources)
def test_builtin_source(name, version):
    sncosmo.get_source(name, version)


@pytest.mark.might_download
@pytest.mark.usefixtures("all_tarfile_errors_are_fatal")
@pytest.mark.parametrize("name", magsystems)
def test_builtin_magsystem(name):
    sncosmo.get_magsystem(name)
