# Licensed under a 3-clause BSD style license - see LICENSES

"""Tests for SALT2Source (and wrapped in Model)"""

import os

import numpy as np
from numpy.testing import assert_allclose, assert_approx_equal
from astropy.tests.helper import remote_data

import sncosmo


def read_header(f):
    """Read header from the open file `f` until first line that doesn't
    start with # or @."""

    meta = {}

    # read header until a line doesn't start with # or @
    while True:
        pos = f.tell()
        line = f.readline()
        if line[0] == '#':
            continue
        elif line[0] == '@':
            key, value = line[1:-1].split()
            meta[key] = float(value)
        else:
            f.seek(pos)
            break

    return meta


@remote_data
def test_salt2source_timeseries_vs_snfit():
    """Test timeseries output from SALT2Source vs pregenerated timeseries
    from snfit (SALT2 software)."""

    source = sncosmo.get_source("salt2", version="2.4")  # fixed version
    model = sncosmo.Model(source)

    dirname = os.path.join(os.path.dirname(__file__), "data")

    for fname in ['salt2_timeseries_1.dat',
                  'salt2_timeseries_2.dat',
                  'salt2_timeseries_3.dat',
                  'salt2_timeseries_4.dat']:
        f = open(os.path.join(dirname, fname), 'r')
        meta = read_header(f)
        time, wave, fluxref = sncosmo.read_griddata_ascii(f)
        f.close()

        # The output from snfit's Salt2Model.SpectrumFlux() has a
        # different definition than sncosmo's model.flux() by a factor
        # of a^2. snfit's definition is the rest-frame flux but at a
        # blue-shifted wavelength.  (There is no correction for photon
        # energy or time dilation. These corrections are made in the
        # integration step.)
        a = 1. / (1. + meta['Redshift'])
        fluxref *= a**2

        model.set(z=meta['Redshift'], t0=meta['DayMax'], x0=meta['X0'],
                  x1=meta['X1'], c=meta['Color'])
        flux = model.flux(time, wave)

        # super good agreement!
        assert_allclose(flux, fluxref, rtol=1e-13)


def test_salt2source_rcov_vs_snfit():
    dirname = os.path.join(os.path.dirname(__file__), "data")

    # read parameters and times
    f = open(os.path.join(dirname, "salt2_rcov_params_times.dat"), 'r')
    meta = read_header(f)
    times = np.loadtxt(f)
    f.close()

    # initialize model and set parameters
    source = sncosmo.get_source("salt2", version="2.4")  # fixed version
    model = sncosmo.Model(source)
    model.set(z=meta['Redshift'], t0=meta['DayMax'], x0=meta['X0'],
              x1=meta['X1'], c=meta['Color'])

    # Test separate bands separately, as thats how they're written to files.
    # (And cross-band covariance is zero.)
    for band in ('SDSSg', 'SDSSr', 'SDSSi'):
        fname = os.path.join(dirname, "salt2_rcov_snfit_{}.dat".format(band))
        ref = np.loadtxt(fname, skiprows=1)

        rcov = model._bandflux_rcov(band, times)
        assert_allclose(ref, rcov, rtol=1.e-5)
