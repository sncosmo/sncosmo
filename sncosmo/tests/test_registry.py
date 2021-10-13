# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test registry functions."""

import numpy as np
import pytest

import sncosmo


def basic_loader(arg, name=None, version=None):
    return arg


def basic_loader_no_version(arg, name=None):
    return arg


@pytest.fixture
def registry():
    # Create a test registry with a few test entries.
    registry = sncosmo._registry.Registry()

    registry.register_loader(
        'test_loader',
        basic_loader,
        args=('test_1',),
        version=1,
    )
    registry.register_loader(
        'test_loader',
        basic_loader,
        args=('test_2',),
        version=2,
        meta={'metaval': 'metatest_2'}
    )
    registry.register('instance_value', 'test_instance')

    return registry


def test_register_bandpass():
    disp = np.array([4000., 4200., 4400., 4600., 4800., 5000.])
    trans = np.array([0., 1., 1., 1., 1., 0.])

    # create a band, register it, make sure we can get it back.
    band = sncosmo.Bandpass(disp, trans, name='tophatg')
    sncosmo.register(band)
    assert sncosmo.get_bandpass('tophatg') is band


def test_register_magsystem():
    magsys = sncosmo.get_magsystem('ab')
    sncosmo.register(magsys, name='test_magsys')
    assert sncosmo.get_magsystem('test_magsys') is magsys


def test_register_source():
    phase = np.linspace(0., 100., 10)
    wave = np.linspace(800., 20000., 100)
    flux = np.ones((len(phase), len(wave)), dtype=float)
    source = sncosmo.TimeSeriesSource(phase, wave, flux, name='test_source')
    sncosmo.register(source)

    # Retrieving a source makes a copy, so just check that the names are the
    # same.
    assert sncosmo.get_source('test_source').name == 'test_source'


def test_register_loader():
    # Register a loader that returns a MagSystem.
    magsys = sncosmo.get_magsystem('ab')
    sncosmo.register_loader(
        sncosmo.magsystems.MagSystem,
        'test_magsystem',
        basic_loader,
        args=(magsys,)
    )

    # make sure that we can get the magnitude system back.
    assert sncosmo.get_magsystem('test_magsystem') is magsys


def test_register_invalid():
    with pytest.raises(ValueError):
        sncosmo.register("invalid")


def test_retrieve():
    sncosmo.registry.retrieve(sncosmo.magsystems.MagSystem, 'ab')


def test_registry_loader_exists(registry):
    with pytest.raises(Exception):
        registry.register_loader(
            'test_loader',
            basic_loader,
            args=('new_test'),
            version=1,
        )


def test_registry_instance_exists(registry):
    with pytest.raises(Exception):
        registry.register('new_instance_value', 'test_instance')


def test_registry_no_name(registry):
    with pytest.raises(ValueError):
        registry.register('asdf')


def test_registry_bad_name(registry):
    class BadClass():
        name = 1
    with pytest.raises(ValueError):
        registry.register(BadClass())


def test_registry_cases(registry):
    # Should work regardless of case.
    for name in ['test_loader', 'TEST_LOADER', 'TesT_LoADeR']:
        registry.retrieve(name)


def test_registry_loader_alias(registry):
    registry.alias('alias_loader', 'test_loader', new_version=1,
                   existing_version=2)
    assert registry.retrieve('alias_loader') == 'test_2'


def test_registry_instance_alias(registry):
    registry.alias('alias_instance', 'test_instance')
    assert registry.retrieve('alias_instance') == 'instance_value'


def test_registry_bad_alias(registry):
    with pytest.raises(Exception):
        registry.alias('alias', 'invalid_key')


def test_registry_retrieve(registry):
    assert registry.retrieve('test_instance') == 'instance_value'


def test_registry_no_version(registry):
    registry.register_loader('noversion_loader', basic_loader_no_version,
                             args=('noversion',))
    assert registry.retrieve('noversion_loader') == 'noversion'


def test_registry_version(registry):
    assert registry.retrieve('test_loader', version=1) == 'test_1'
    assert registry.retrieve('test_loader', version=2) == 'test_2'


def test_registry_default_version(registry):
    assert registry.retrieve('test_loader') == 'test_2'


def test_registry_missing(registry):
    with pytest.raises(Exception):
        assert registry.retrieve('missing')


def test_registry_missing_version(registry):
    with pytest.raises(Exception):
        assert registry.retrieve('test_loader', version=3)


def test_registry_loaders_metadata(registry):
    meta = registry.get_loaders_metadata()
    assert len(meta) == 2
