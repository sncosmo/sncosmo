# Licensed under a 3-clause BSD style license - see LICENSES
from __future__ import print_function

from os.path import join, dirname

import sncosmo


def test_read_snana_ascii():
    fname = join(dirname(__file__), "data", "snana_ascii_example.dat")
    meta, tables = sncosmo.read_snana_ascii(fname, default_tablename="OBS")

    # only 1 table
    assert len(tables) == 1
    data = tables["OBS"]

    assert len(data) == 4  # 4 rows.
    assert len(data.colnames) == 13  # 13 columns.
