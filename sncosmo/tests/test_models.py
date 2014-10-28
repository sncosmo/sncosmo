# Licensed under a 3-clause BSD style license - see LICENSES

from cStringIO import StringIO

import numpy as np
from numpy.testing import assert_allclose, assert_approx_equal

import sncosmo
from sncosmo import registry


def flatsource():
    """Create and return a TimeSeriesSource with a flat spectrum == 1.0 at
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
        assert_approx_equal(ans, f)

    def test_bandflux_shapes(self):
        # Just check that these work.
        self.source.bandflux("bessellb", 0., zp=25., zpsys="ab")
        self.source.bandflux("bessellb", [0.1, 0.2], zp=25., zpsys="ab")
        self.source.bandflux(["bessellb", "bessellv"], [0.1, 0.2], zp=25.,
                             zpsys="ab")


class TestSALT2Source:
    def setup_class(self):
        """Create a SALT2 model with a lot of components set to 1."""

        phase = np.linspace(0., 100., 10)
        wave = np.linspace(1000., 10000., 100)
        vals1d = np.zeros(len(phase), dtype=np.float64)

        # Create some 2-d grid files
        files = []
        for i in [0, 1]:
            f = StringIO()
            sncosmo.write_griddata_ascii(phase, wave, vals, f)
            f.seek(0)  # return to start of file.
            files.append(f)

        # CL file. The CL in magnitudes will be
        # CL(wave) = -(wave - B) / (V - B)  [B = 4302.57, V = 5428.55]
        # and transmission will be 10^(-0.4 * CL(wave))^c
        clfile = StringIO()
        clfile.write("1\n"
                     "0.0\n"
                     "Salt2ExtinctionLaw.version 1\n"
                     "Salt2ExtinctionLaw.min_lambda 3000\n"
                     "Salt2ExtinctionLaw.max_lambda 7000\n")
        clfile.seek(0)

        # Create some more 2-d grid files
        for factor in [1., 0.01, 0.01, 0.01]:
            f = StringIO()
            sncosmo.write_griddata_ascii(phase, wave, factor * vals, f)
            f.seek(0)  # return to start of file.
            files.append(f)

        # Create a 1-d grid file (color dispersion)
        cdfile = StringIO()
        for w in wave:
            cdfile.write("{0:f} {1:f}\n".format(w, 0.2))
        cdfile.seek(0)  # return to start of file.

        # Create a SALT2Source
        self.source = sncosmo.SALT2Source(m0file=files[0],
                                          m1file=files[1],
                                          clfile=clfile,
                                          errscalefile=files[2],
                                          lcrv00file=files[3],
                                          lcrv11file=files[4],
                                          lcrv01file=files[5],
                                          cdfile=cdfile)

        def test_bandflux_rcov(self):

            # component 1:
            # ans = (F0/F1)^2 S^2 (V00 + 2 x1 V01 + x1^2 V11)
            # when x1=0, this reduces to S^2 V00 = 1^2 * 0.01 = 0.01
            #
            # component 2:
            # cd^2 = 0.04

            band = ['bessellb', 'bessellb', 'bessellr', 'bessellr',
                    'besselli']
            phase = [10., 20., 30., 40., 50.]
            self.source.set(x1=0.0)
            result = self.source.bandflux_rcov(band, phase)
            expected = np.array([[0.05, 0.04, 0.,   0.,   0.],
                                 [0.04, 0.05, 0.,   0.,   0.],
                                 [0.,   0.,   0.05, 0.04, 0.],
                                 [0.,   0.,   0.04, 0.05, 0.],
                                 [0.,   0.,   0.,   0.,   0.05]])
            assert_allclose(result, expected)


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
