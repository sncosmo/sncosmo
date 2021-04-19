# Licensed under a 3-clause BSD style license - see LICENSES

import os
from io import BytesIO, StringIO
from os.path import dirname, join
from tempfile import NamedTemporaryFile, mkdtemp

import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from numpy.testing import assert_allclose

import sncosmo

# Dummy data used for read_lc/write_lc round-tripping tests
time = [1., 2., 3., 4.]
band = ['sdssg', 'sdssr', 'sdssi', 'sdssz']
zp = [25., 25., 25., 25.]
zpsys = ['ab', 'ab', 'ab', 'ab']
flux = [1., 1., 1., 1.]
fluxerr = [0.1, 0.1, 0.1, 0.1]
lcdata = Table(data=(time, band, flux, fluxerr, zp, zpsys),
               names=('time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'),
               meta={'a': 1, 'b': 1.0, 'c': 'one'})


def test_read_griddata_ascii():

    # Write a temporary test file.
    f = StringIO()
    f.write("0. 0. 0.\n"
            "0. 1. 0.\n"
            "0. 2. 0.\n"
            "1. 0. 0.\n"
            "1. 1. 0.\n"
            "1. 2. 0.\n")
    f.seek(0)

    x0, x1, y = sncosmo.read_griddata_ascii(f)
    f.close()

    assert_allclose(x0, np.array([0., 1.]))
    assert_allclose(x1, np.array([0., 1., 2.]))


def test_write_griddata_ascii():

    x0 = np.array([0., 1.])
    x1 = np.array([0., 1., 2.])
    y = np.zeros((2, 3))

    f = StringIO()
    sncosmo.write_griddata_ascii(x0, x1, y, f)

    # Read it back
    f.seek(0)
    x0_in, x1_in, y_in = sncosmo.read_griddata_ascii(f)
    f.close()
    assert_allclose(x0_in, x0)
    assert_allclose(x1_in, x1)
    assert_allclose(y_in, y)

    # with a filename:
    dirname = mkdtemp()
    fname = os.path.join(dirname, 'griddata.dat')
    sncosmo.write_griddata_ascii(x0, x1, y, fname)
    x0_in, x1_in, y_in = sncosmo.read_griddata_ascii(fname)
    assert_allclose(x0_in, x0)
    assert_allclose(x1_in, x1)
    assert_allclose(y_in, y)
    os.remove(fname)
    os.rmdir(dirname)


def test_griddata_fits():
    """Round tripping with write_griddata_fits() and read_griddata_fits()"""

    x0 = np.array([0., 1.])
    x1 = np.array([0., 1., 2.])
    y = np.zeros((2, 3))

    f = BytesIO()
    sncosmo.write_griddata_fits(x0, x1, y, f)

    # Read it back
    f.seek(0)
    x0_in, x1_in, y_in = sncosmo.read_griddata_fits(f)
    assert_allclose(x0_in, x0)
    assert_allclose(x1_in, x1)
    assert_allclose(y_in, y)
    f.close()

    # Test reading 3-d grid data. We don't have a writer for
    # this, so we write a temporary FITS file by hand.
    x2 = np.array([3., 5., 7., 9])
    y = np.zeros((len(x0), len(x1), len(x2)))

    # write a FITS file that represents x0, x1, x2, y
    w = wcs.WCS(naxis=3)
    w.wcs.crpix = [1, 1, 1]
    w.wcs.crval = [x2[0], x1[0], x0[0]]
    w.wcs.cdelt = [2., 1., 1.]
    hdu = fits.PrimaryHDU(y, header=w.to_header())
    f = BytesIO()
    hdu.writeto(f)

    # Read it back
    f.seek(0)
    x0_in, x1_in, x2_in, y_in = sncosmo.read_griddata_fits(f)
    f.close()

    assert_allclose(x0_in, x0)
    assert_allclose(x1_in, x1)
    assert_allclose(x2_in, x2)
    assert_allclose(y_in, y)


def test_read_lc():
    f = StringIO("""
@id 1
@RA 36.0
@description good
time band flux fluxerr zp zpsys
50000. g 1. 0.1 25. ab
50000.1 r 2. 0.1 25. ab
""")
    t = sncosmo.read_lc(f, format='ascii')
    assert str(t) == ("  time  band flux fluxerr  zp  zpsys\n"
                      "------- ---- ---- ------- ---- -----\n"
                      "50000.0    g  1.0     0.1 25.0    ab\n"
                      "50000.1    r  2.0     0.1 25.0    ab")
    assert t.meta['id'] == 1
    assert t.meta['RA'] == 36.0
    assert t.meta['description'] == 'good'


def test_read_salt2():
    fname = join(dirname(__file__), "data", "lc-03D4ag.list")
    data = sncosmo.read_lc(fname, format="salt2")

    # Test a few columns
    assert_allclose(data["Date"][0:4],
                    [52816.54, 52824.59, 52851.53, 52873.4])
    assert_allclose(data["ZP"][0:4], 27.036167)

    assert np.all(data["Filter"][0:4] == "MEGACAMPSF::g")
    assert np.all(data["MagSys"] == "AB_B12")

    # Test a bit of metadata
    assert_allclose(data.meta["Z_HELIO"], 0.285)
    assert_allclose(data.meta["RA"], 333.690959)
    assert data.meta["z_source"] == "H"


def test_read_salt2_cov():
    fname = join(dirname(__file__), "data", "lc-03D4ag.list")
    data = sncosmo.read_lc(fname, format="salt2", read_covmat=True)
    assert data["Fluxcov"].shape == (len(data), len(data))
    assert_allclose(data["Fluxcov"][0:3, 0:3],
                    [[0.867712297284, 0.01139998771, 0.01119398747],
                     [0.01139998771, 2.03512047975, 0.01190299234],
                     [0.01119398747, 0.01190299234, 1.3663344852]])


def test_read_salt2_old():
    dname = join(dirname(__file__), "data", "SNLS3-04D3gx")
    data = sncosmo.read_lc(dname, format="salt2-old")

    # Test length and column names:
    assert len(data) == 25 + 37 + 38 + 18  # g + r + i + z lengths
    assert data.colnames == ["Date", "Flux", "Fluxerr", "ZP", "Filter",
                             "MagSys"]

    # Test a bit of metadata and data
    assert data.meta["NAME"] == "04D3gx"
    assert_allclose(data.meta["Redshift"], 0.91)
    assert_allclose(data.meta["RA"], 215.056948)
    assert np.all(data["MagSys"] == "VEGA")


def test_roundtripping():
    for format in ['json', 'ascii', 'salt2']:
        f = NamedTemporaryFile(delete=False)
        f.close()  # close to ensure that we can open it in write_lc()

        # raw=True is for the benefit of salt2 writer that modifies column
        # and header names by default.
        sncosmo.write_lc(lcdata, f.name, format=format, raw=True,
                         pedantic=False)
        data = sncosmo.read_lc(f.name, format=format)

        for key in lcdata.colnames:
            assert np.all(data[key] == lcdata[key])
        for key in lcdata.meta:
            assert data.meta[key] == lcdata.meta[key]

        os.unlink(f.name)


def test_write_lc_salt2():
    """Extra test to see if column renaming works"""
    f = NamedTemporaryFile(delete=False)
    f.close()  # close to ensure that we can open it in write_lc()
    sncosmo.write_lc(lcdata, f.name, format='salt2')
    os.unlink(f.name)


def test_write_lc_snana():
    """Just check if the snana writer works without error."""
    f = NamedTemporaryFile(delete=False)
    f.close()  # close to ensure that we can open it in write_lc()
    sncosmo.write_lc(lcdata, f.name, format='snana', pedantic=False)
    os.unlink(f.name)


def test_load_example_data():
    data = sncosmo.load_example_data()

def test_load_example_spectrum_data():
    wave, flux, fluxerr = sncosmo.load_example_spectrum_data()

    assert len(wave) > 0
    assert len(wave) == len(flux) == len(fluxerr)