import numpy as np
import pytest

from astropy.table import Table
from numpy.testing import assert_allclose

import sncosmo


try:
    import iminuit

    HAS_IMINUIT = True

except ImportError:
    HAS_IMINUIT = False


def test_bin_edges_linear():
    """Ensure that we can recover consistent bin edges for a spectrum from bin centers.

    Internally, the bin edges are stored rather than the bin centers.
    """
    wave = np.linspace(3000, 8000, 100)
    flux = np.ones_like(wave)
    spec = sncosmo.Spectrum(wave, flux)

    assert_allclose(wave, spec.wave, rtol=1.e-5)


def test_bin_edges_log():
    """Ensure that we can recover consistent bin edges for a spectrum from bin centers.

    Internally, the bin edges are stored rather than the bin centers.
    """
    wave = np.logspace(np.log10(3000), np.log10(8000), 100)
    flux = np.ones_like(wave)
    spec = sncosmo.Spectrum(wave, flux)

    assert_allclose(wave, spec.wave, rtol=1.e-5)


class TestSpectrum:
    def setup_class(self):
        # Simulate a spectrum
        model = sncosmo.Model(source='hsiao-subsampled')
        params = {'t0': 10., 'amplitude': 1.e-7, 'z': 0.2}
        start_params = {'t0': 0., 'amplitude': 1., 'z': 0.}
        model.set(**params)

        # generate a fake spectrum with no errors. note: we simulate a high resolution
        # spectrum and then bin it up. we also include large covariance between spectral
        # elements to verify that we are handling covariance properly.
        spec_time = params['t0'] + 5.
        sim_wave = np.arange(3000, 9000)
        sim_flux = model.flux(spec_time, sim_wave)
        sim_fluxcov = 0.01 * np.max(sim_flux)**2 * np.ones((len(sim_flux),
                                                            len(sim_flux)))
        sim_fluxcov += np.diag(0.1 * sim_flux**2)
        spectrum = sncosmo.Spectrum(sim_wave, sim_flux, fluxcov=sim_fluxcov,
                                    time=spec_time)

        # generate a binned up low-resolution spectrum.
        bin_wave = np.linspace(3500, 8500, 200)
        bin_spectrum = spectrum.rebin(bin_wave)

        # generate fake photometry with no errors
        points_per_band = 12
        bands = points_per_band * ['bessellux', 'bessellb', 'bessellr',
                                   'besselli']
        times = params['t0'] + np.linspace(-10., 60., len(bands))
        zp = len(bands) * [25.]
        zpsys = len(bands) * ['ab']
        flux = model.bandflux(bands, times, zp=zp, zpsys=zpsys)
        fluxerr = len(bands) * [0.1 * np.max(flux)]
        photometry = Table({
            'time': times,
            'band': bands,
            'flux': flux,
            'fluxerr': fluxerr,
            'zp': zp,
            'zpsys': zpsys
        })

        self.model = model
        self.photometry = photometry
        self.spectrum = spectrum
        self.bin_spectrum = bin_spectrum
        self.params = params
        self.start_params = start_params

    def test_bandflux(self):
        """Check synthetic photometry.

        We compare synthetic photometry on high and low resolution spectra. It should
        stay the same.
        """
        bandflux_highres = self.spectrum.bandflux('sdssg')
        bandflux_lowres = self.bin_spectrum.bandflux('sdssg')

        assert_allclose(bandflux_highres, bandflux_lowres, rtol=1.e-3)

    def test_bandflux_multi(self):
        """Check synthetic photometry with multiple bands."""
        bands = ['sdssg', 'sdssr', 'sdssi']
        bandflux_highres = self.spectrum.bandflux(bands)
        bandflux_lowres = self.bin_spectrum.bandflux(bands)

        assert_allclose(bandflux_highres, bandflux_lowres, rtol=1.e-3)

    def test_bandflux_zpsys(self):
        """Check synthetic photometry with a magnitude system."""
        bands = ['sdssg', 'sdssr', 'sdssi']
        bandflux_highres = self.spectrum.bandflux(bands, 25., 'ab')
        bandflux_lowres = self.spectrum.bandflux(bands, 25., 'ab')

        assert_allclose(bandflux_highres, bandflux_lowres, rtol=1.e-3)

    def test_bandfluxcov(self):
        """Check synthetic photometry with covariance."""
        bands = ['sdssg', 'sdssr', 'sdssi']
        flux_highres, cov_highres = self.spectrum.bandfluxcov(bands)
        flux_lowres, cov_lowres = self.bin_spectrum.bandfluxcov(bands)

        assert_allclose(flux_highres, flux_lowres, rtol=1.e-3)
        assert_allclose(cov_highres, cov_lowres, rtol=1.e-3)

    def test_bandmag(self):
        """Check synthetic photometry in magnitudes."""
        bands = ['sdssg', 'sdssr', 'sdssi']
        bandmag_highres = self.spectrum.bandmag(bands, 'ab')
        bandmag_lowres = self.bin_spectrum.bandmag(bands, 'ab')

        assert_allclose(bandmag_highres, bandmag_lowres, rtol=1.e-3)

    @pytest.mark.skipif('not HAS_IMINUIT')
    def test_fit_lc_spectra(self):
        """Check fit results for a single high-resolution spectrum."""
        self.model.set(**self.start_params)
        res, fitmodel = sncosmo.fit_lc(model=self.model, spectra=self.bin_spectrum,
                                       vparam_names=['amplitude', 'z', 't0'],
                                       bounds={'z': (0., 0.3)})

        # set model to true parameters and compare to fit results.
        self.model.set(**self.params)
        assert_allclose(res.parameters, self.model.parameters, rtol=1.e-3)

    @pytest.mark.skipif('not HAS_IMINUIT')
    def test_fit_lc_both(self):
        """Check fit results for both spectra and photometry."""
        self.model.set(**self.start_params)
        res, fitmodel = sncosmo.fit_lc(self.photometry, model=self.model,
                                       spectra=self.bin_spectrum,
                                       vparam_names=['amplitude', 'z', 't0'],
                                       bounds={'z': (0., 0.3)})

        # set model to true parameters and compare to fit results.
        self.model.set(**self.params)
        assert_allclose(res.parameters, self.model.parameters, rtol=1.e-3)

    @pytest.mark.skipif('not HAS_IMINUIT')
    def test_fit_lc_multiple_spectra(self):
        """Check fit results for multiple spectra."""
        self.model.set(**self.start_params)
        res, fitmodel = sncosmo.fit_lc(model=self.model,
                                       spectra=[self.bin_spectrum, self.bin_spectrum],
                                       vparam_names=['amplitude', 'z', 't0'],
                                       bounds={'z': (0., 0.3)})

        # set model to true parameters and compare to fit results.
        self.model.set(**self.params)
        assert_allclose(res.parameters, self.model.parameters, rtol=1.e-3)
