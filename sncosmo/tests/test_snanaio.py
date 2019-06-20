# Licensed under a 3-clause BSD style license - see LICENSES

from os.path import join, dirname

from numpy.testing import assert_allclose

import sncosmo


def test_read_snana_ascii():
    fname = join(dirname(__file__), "data", "snana_ascii_example.dat")
    meta, tables = sncosmo.read_snana_ascii(fname, default_tablename="OBS")

    # test a few different types of metadata
    assert meta['SURVEY'] == 'DES'
    assert meta['FAKE'] == 0
    assert_allclose(meta['REDSHIFT_HELIO'], 0.3614)

    # only 1 table
    assert len(tables) == 1
    data = tables["OBS"]

    assert len(data) == 4  # 4 rows.
    assert len(data.colnames) == 13  # 13 columns.


def test_read_snana_fits():
    fname1 = join(dirname(__file__), "data", "snana_fits_example_head.fits")
    fname2 = join(dirname(__file__), "data", "snana_fits_example_phot.fits")
    sne = sncosmo.read_snana_fits(fname1, fname2)
    assert len(sne) == 2


def test_read_snana_simlib():
    fname = join(dirname(__file__), "data", "snana_simlib_example.dat")
    meta, obs_sets = sncosmo.read_snana_simlib(fname)
    assert len(obs_sets) == 2


def test_read_snana_simlib_noend():
    fname = join(dirname(__file__), "data", "snana_simlib_example_noend.dat")
    meta, obs_sets = sncosmo.read_snana_simlib(fname)
    assert len(obs_sets) == 2
