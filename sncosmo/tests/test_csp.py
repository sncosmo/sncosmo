
import sncosmo
from numpy.testing import assert_allclose
from astropy.utils.data import get_pkg_data_filename

def test_csp_magsys_calibration():
    
    csp = sncosmo.get_magsystem('csp')
    csp_info_path = get_pkg_data_filename('data/csp/csp_filter_info.dat')

    # read it into a numpy array
    csp_filter_data = np.genfromtxt(csp_info_path, names=True, dtype=None,
                                    skip_header=3)

    answers = csp_filter_data['natural_mag']
    bands   = csp_filter_data['name']
    
    for band, answer in zip(bands, answers):
        assert_allclose(csp.band_mag_to_flux(csp.standard_mag(band)), answer)
        

        
        
