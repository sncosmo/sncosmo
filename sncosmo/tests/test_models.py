# Licensed under a 3-clause BSD style license - see LICENSES

import numpy as np
from numpy.testing import assert_allclose

import sncosmo
from sncosmo import registry

class TestTimeSeriesSource:
    def setup_class(self):
        # Create a TimeSeriesSource with a flat spectrum at all times.
        phase = np.linspace(0., 100., 10)
        wave = np.linspace(1000., 10000., 100)
        flux = np.ones((len(phase), len(wave)), dtype=np.float)
        self.source = sncosmo.TimeSeriesSource(phase, wave, flux)

    def test_flux(self):
        for a in [1., 2.]:
            self.source.set(amplitude=a)
            assert_allclose(self.source.flux(1., 2000.), a)
            assert_allclose(self.source.flux(1., [2000.]), np.array([a]))
            assert_allclose(self.source.flux([1.], 2000.), np.array([[a]]))
            assert_allclose(self.source.flux([1.], [2000.]), np.array([[a]]))

    def test_getsource(self):
        # register the source & retrieve it from the registry
        registry.register(self.source, name="testsource",
                          data_class=sncosmo.Source)
        m = sncosmo.get_source("testsource")

    def test_bandflux(self):
        self.source.set(amplitude=1.0)
        f = self.source.bandflux("bessellb", 0.)

        # Correct answer
        b = sncosmo.get_bandpass("bessellb")
        ans = np.sum(b.trans * b.wave * b.dwave) / sncosmo.models.HC_ERG_AA
        assert ans == f

    def test_bandflux_shapes(self):
        # Just check that these work.
        self.source.bandflux("bessellb", 0., zp=25., zpsys="ab")
        self.source.bandflux("bessellb", [0.1, 0.2], zp=25., zpsys="ab")
        self.source.bandflux(["bessellb", "bessellv"], [0.1, 0.2], zp=25.,
                             zpsys="ab")
