import pytest

import sncosmo

from sncosmo.bandpasses import _BANDPASSES, _BANDPASS_INTERPOLATORS
from sncosmo.magsystems import _MAGSYSTEMS
from sncosmo.models import _SOURCES


bandpasses = [i['name'] for i in _BANDPASSES.get_loaders_metadata()]
bandpass_interpolators = [i['name'] for i in
                          _BANDPASS_INTERPOLATORS.get_loaders_metadata()]
magsystems = [i['name'] for i in _MAGSYSTEMS.get_loaders_metadata()]
sources = [(i['name'], i['version']) for i in _SOURCES.get_loaders_metadata()]


@pytest.mark.might_download
@pytest.mark.parametrize("name", bandpasses)
def test_builtin_bandpass(name):
    sncosmo.get_bandpass(name)


@pytest.mark.might_download
@pytest.mark.parametrize("name", bandpass_interpolators)
def test_builtin_bandpass_interpolator(name):
    interpolator = _BANDPASS_INTERPOLATORS.retrieve(name)
    interpolator.at(interpolator.minpos())


@pytest.mark.might_download
@pytest.mark.parametrize("name,version", sources)
def test_builtin_source(name, version):
    sncosmo.get_source(name, version)


@pytest.mark.might_download
@pytest.mark.parametrize("name", magsystems)
def test_builtin_magsystem(name):
    sncosmo.get_magsystem(name)
