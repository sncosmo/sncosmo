# Licensed under a 3-clause BSD style license - see LICENSES
from __future__ import print_function


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
