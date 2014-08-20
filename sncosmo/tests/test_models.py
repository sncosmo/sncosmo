# Licensed under a 3-clause BSD style license - see LICENSES

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

import sncosmo
from sncosmo import registry

def flatsource():
    """Create and return a TimeSeriesSource with a flat spectrum at
    all times."""
    phase = np.linspace(0., 100., 10)
    wave = np.linspace(800., 20000., 100)
    flux = np.ones((len(phase), len(wave)), dtype=np.float)
    return sncosmo.TimeSeriesSource(phase, wave, flux)

class TestTimeSeriesSource:
    def setup_class(self):
        self.source = flatsource()

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
        assert_almost_equal(ans, f)

    def test_bandflux_shapes(self):
        # Just check that these work.
        self.source.bandflux("bessellb", 0., zp=25., zpsys="ab")
        self.source.bandflux("bessellb", [0.1, 0.2], zp=25., zpsys="ab")
        self.source.bandflux(["bessellb", "bessellv"], [0.1, 0.2], zp=25.,
                             zpsys="ab")


class TestModel:
    def setup_class(self):
        self.model = sncosmo.Model(source=flatsource(),
                                   effects=[sncosmo.CCM89Dust()],
                                   effect_frames=['obs'],
                                   effect_names=['mw'])
    
    def test_minwave(self):
        
        # at redshift zero, should be determined by effect minwave (~909)
        self.model.set(z=0.)
        ans = max(self.model.source.minwave(),
                  self.model.effects[0].minwave())
        assert self.model.minwave() == ans

        # at redshift 1, should be determined by effect minwave (800*2)
        z = 1.
        self.model.set(z=z)
        ans = max(self.model.source.minwave() * (1.+z),
                  self.model.effects[0].minwave())
        assert self.model.minwave() == ans

    def test_maxwave(self):
        
        # at redshift zero, should be determined by source maxwave (20000)
        self.model.set(z=0.)
        ans = min(self.model.source.maxwave(),
                  self.model.effects[0].maxwave())
        assert self.model.maxwave() == ans

        # at redshift 1, should be determined by effect maxwave (33333)
        z = 1.
        self.model.set(z=z)
        ans = min(self.model.source.maxwave() * (1.+z),
                  self.model.effects[0].maxwave())
        assert self.model.maxwave() == ans

    def test_set_source_peakabsmag(self):
        
        # Both Bandpass and str should work
        band = sncosmo.get_bandpass('desg')
        self.model.set_source_peakabsmag(-19.3, 'desg', 'ab')
        self.model.set_source_peakabsmag(-19.3, band, 'ab')
