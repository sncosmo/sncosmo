# Licensed under a 3-clause BSD style license - see LICENSES

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from astropy.table import Table

import sncosmo

class TestFitting:
    def setup_class(self):
        model = sncosmo.Model(source='hsiao-subsampled')
        params = {'t0': 56000., 'amplitude': 1.e-7, 'z': 0.1}
        
        # generate fake data with no errors
        points_per_band = 12
        bands = points_per_band * ['desg', 'desr', 'desi', 'desz']
        times = params['t0'] + np.linspace(-10., 60., len(bands))
        zp = len(bands) * [25.]
        zpsys = len(bands) * ['ab']
        model.set(**params)
        flux = model.bandflux(bands, times, zp=zp, zpsys=zpsys)
        fluxerr = len(bands) * [0.1 * np.max(flux)]
        data = Table({'time': times,
                      'band': bands,
                      'flux': flux,
                      'fluxerr': fluxerr,
                      'zp': zp,
                      'zpsys': zpsys})

        # reset parameters
        model.set(z=0., t0=0., amplitude=1.)

        self.model = model
        self.data = data
        self.params = params

    def test_fit_lc(self):
        res, fitmodel = sncosmo.fit_lc(self.data, self.model,
                                       ['z', 't0', 'amplitude'],
                                       bounds={'z':(0., 1.0)})

        # set model to true parameters
        self.model.set(**self.params)
        assert_allclose(res.parameters, self.model.parameters, rtol=1.e-3)
