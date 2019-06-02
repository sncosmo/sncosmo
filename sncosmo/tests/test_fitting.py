# Licensed under a 3-clause BSD style license - see LICENSES

from os.path import dirname, join

import pytest
import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_almost_equal
from astropy.table import Table

import sncosmo

try:
    import iminuit
    HAS_IMINUIT = True
except ImportError:
    HAS_IMINUIT = False

try:
    import nestle
    HAS_NESTLE = True
except ImportError:
    HAS_NESTLE = False


class TestFitting:
    def setup_class(self):
        model = sncosmo.Model(source='hsiao-subsampled')
        params = {'t0': 56000., 'amplitude': 1.e-7, 'z': 0.2}

        # generate fake data with no errors
        points_per_band = 12
        bands = points_per_band * ['bessellux', 'bessellb', 'bessellr',
                                   'besselli']
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

    @pytest.mark.skipif('not HAS_IMINUIT')
    def test_fit_lc(self):
        """Ensure that fit results match input model parameters (data are
        noise-free).

        Pass in parameter names in order different from that stored in
        model; tests parameter re-ordering."""
        res, fitmodel = sncosmo.fit_lc(self.data, self.model,
                                       ['amplitude', 'z', 't0'],
                                       bounds={'z': (0., 1.0)})

        # set model to true parameters and compare to fit results.
        self.model.set(**self.params)
        assert_allclose(res.parameters, self.model.parameters, rtol=1.e-3)

    @pytest.mark.skipif('not HAS_IMINUIT')
    def test_wrong_param_names(self):
        """Supplying parameter names that are not part of the model should
        raise an error."""

        # a parameter not in the model
        with pytest.raises(ValueError):
            res, fitmodel = sncosmo.fit_lc(self.data, self.model,
                                           ['t0', 'not_a_param'])

        # no parameters
        with pytest.raises(ValueError):
            res, fitmodel = sncosmo.fit_lc(self.data, self.model, [])

    @pytest.mark.skipif('not HAS_NESTLE')
    def test_nest_lc(self):
        """Ensure that nested sampling runs.

        Pass in parameter names in order different from that stored in
        model; tests parameter re-ordering.
        """

        rstate = RandomState(0)

        self.model.set(**self.params)

        res, fitmodel = sncosmo.nest_lc(
            self.data, self.model, ['amplitude', 'z', 't0'],
            bounds={'z': (0., 1.0)}, guess_amplitude_bound=True, npoints=50,
            rstate=rstate)

        assert_allclose(fitmodel.parameters, self.model.parameters, rtol=0.05)


@pytest.mark.might_download
@pytest.mark.skipif('not HAS_IMINUIT')
def test_fit_lc_vs_snfit():
    """Test fit_lc versus snfit result for one SN."""

    # purposefully use CCM dust to match snfit
    model = sncosmo.Model(source='salt2',
                          effects=[sncosmo.CCM89Dust()],
                          effect_names=['mw'],
                          effect_frames=['obs'])

    fname = join(dirname(__file__), "data", "lc-03D4ag.list")

    data = sncosmo.read_lc(fname, format='salt2', read_covmat=True,
                           expand_bands=True)
    model.set(mwebv=data.meta['MWEBV'], z=data.meta['Z_HELIO'])
    result, fitted_model = sncosmo.fit_lc(
        data, model, ['t0', 'x0', 'x1', 'c'],
        bounds={'x1': (-3., 3.), 'c': (-0.4, 0.4)},
        modelcov=True,
        phase_range=(-15., 45.),
        wave_range=(3000., 7000.),
        warn=False,
        verbose=False)

    print(result)
    assert result.ndof == 25
    assert result.nfit == 3
    assert_allclose(fitted_model['t0'], 52830.9313, atol=0.01, rtol=0.)
    assert_allclose(fitted_model['x0'], 5.6578663e-05, atol=0., rtol=0.005)
    assert_allclose(fitted_model['x1'], 0.937399344, atol=0.005, rtol=0.)
    assert_allclose(fitted_model['c'], -0.0851965244, atol=0.001, rtol=0.)

    # errors
    assert_allclose(result.errors['t0'], 0.0955792638, atol=0., rtol=0.01)
    assert_allclose(result.errors['x0'], 1.52745001e-06, atol=0., rtol=0.01)
    assert_allclose(result.errors['x1'], 0.104657847, atol=0., rtol=0.01)
    assert_allclose(result.errors['c'], 0.0234763446, atol=0., rtol=0.01)
