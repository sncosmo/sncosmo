# Licensed under a 3-clause BSD style license - see LICENSES

from copy import deepcopy
from os.path import dirname, join
from collections import defaultdict

import numpy as np
import pytest
from astropy.table import Table
from numpy.random import RandomState
from numpy.testing import assert_allclose

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

try:
    import emcee

    HAS_EMCEE = True

except ImportError:
    HAS_EMCEE = False


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
        data = Table({
            'time': times,
            'band': bands,
            'flux': flux,
            'fluxerr': fluxerr,
            'zp': zp,
            'zpsys': zpsys
        })

        # generate fake spectral time series data with no errors
        wave = np.geomspace(model.minwave()+1, model.maxwave()-1, 300)
        times = np.arange(model.mintime(), model.maxtime(), 3)
        sts_data = defaultdict(list)
        for time in times:
            flux = model.flux(time, wave)
            flux_err = 0.1 * np.max(flux) * np.ones(len(flux))
            sts_data['time'].append(time * np.ones(len(flux)))
            sts_data['wave'].append(wave)
            sts_data['flux'].append(flux)
            sts_data['flux_err'].append(flux_err)

        for key, val in sts_data.items():
            sts_data[key] = np.array(val).flatten()
        sts_data = Table(sts_data)

        # reset parameters
        model.set(z=0., t0=0., amplitude=1.)

        self.model = model
        self.data = data
        self.sts_data = sts_data
        self.params = params

    def _test_mutation(self, fit_func):
        """Test a fitting function does not mutate arguments"""

        # Some fitting functions require bounds for all varied parameters
        bounds = {}
        for param, param_val in self.params.items():
            bounds[param] = (param_val * .95, param_val * 1.05)

        # Preserve original input data
        vparams = list(self.params.keys())
        test_data = deepcopy(self.data)
        test_model = deepcopy(self.model)
        test_bounds = deepcopy(bounds)
        test_vparams = deepcopy(vparams)

        # Check for argument mutation
        fit_func(test_data, test_model, test_vparams, bounds=test_bounds)
        param_preserved = all(a == b for a, b in zip(vparams, test_vparams))
        model_preserved = all(
            a == b for a, b in
            zip(self.model.parameters, test_model.parameters)
        )

        err_msg = '``{}`` argument was mutated'
        assert all(self.data == test_data), err_msg.format('data')
        assert bounds == test_bounds, err_msg.format('bounds')
        assert param_preserved, err_msg.format('vparam_names')
        assert model_preserved, err_msg.format('model')

    @pytest.mark.skipif('not HAS_IMINUIT')
    def test_fitlc_arg_mutation(self):
        """Test ``fit_lc`` does not mutate it's arguments"""

        self._test_mutation(sncosmo.fit_lc)

    @pytest.mark.skipif('not HAS_NESTLE')
    def test_nestlc_arg_mutation(self):
        """Test ``nest_lc`` does not mutate it's arguments"""

        self._test_mutation(sncosmo.nest_lc)

    @pytest.mark.skipif('not HAS_EMCEE')
    def test_mcmclc_arg_mutation(self):
        """Test ``mcmc_lc`` does not mutate it's arguments"""

        self._test_mutation(sncosmo.mcmc_lc)

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
    def test_fit_sts(self):
        """Ensure that spectral time series fit results match model parameters
        when data are noise-free.

        Note that redshift fitting is not yet implemented"""
        res, fitmodel = sncosmo.fit_sts(self.sts_data, self.model,
                                        ['amplitude', 't0'])
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
