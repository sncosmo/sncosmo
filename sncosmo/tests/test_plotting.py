# Licensed under a 3-clause BSD style license - see LICENSES

import numpy as np
import pytest
import sncosmo
try:
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@pytest.mark.skipif('not HAS_MATPLOTLIB')
class TestPlotLC:
    def setup_class(self):
        # Create a TimeSeriesSource with a flat spectrum at all times.
        phase = np.linspace(0., 100., 10)
        wave = np.linspace(1000., 10000., 100)
        flux = np.ones((len(phase), len(wave)), dtype=np.float)
        source = sncosmo.TimeSeriesSource(phase, wave, flux)
        self.model = sncosmo.Model(source=source)

    def test_plotmodel(self):
        fig = sncosmo.plot_lc(model=self.model, bands=['bessellb', 'bessellr'])
        assert isinstance(fig, Figure)
