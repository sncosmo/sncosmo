# Licensed under a 3-clause BSD style license - see LICENSES

import numpy as np
from numpy.testing import assert_allclose
import sncosmo

class TestTimeSeriesModel:
    def setup_class(self):
        phase = np.linspace(0., 100., 10)
        wave = np.linspace(1000., 10000., 100)
        flux = np.ones((len(phase), len(wave)), dtype=np.float)
        self.model = sncosmo.TimeSeriesModel(phase, wave, flux)

    def test_flux(self):
        for a in [1., 2.]:
            self.model.set(amplitude=a)
            assert_allclose(self.model.flux(1., 2000.), a)
            assert_allclose(self.model.flux(1., [2000.]), np.array([a]))
            assert_allclose(self.model.flux([1.], 2000.), np.array([[a]]))
            assert_allclose(self.model.flux([1.], [2000.]), np.array([[a]]))
