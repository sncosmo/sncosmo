
import sncosmo
from numpy.testing import assert_allclose
from astropy.utils.data import get_pkg_data_filename
import numpy as np

def test_natmag_bandfail():
    
    csp = sncosmo.get_magsystem('csp')
    try:
        csp.zpbandflux('desi')
    except ValueError:
        assert True
    else:
        assert False
