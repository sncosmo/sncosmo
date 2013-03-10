# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Importing this module registers loaders for built-in magnitude systems.
# The module docstring, a table of the magnitude systems, is generated at the
# after all the magnitude systems are registered.

from astropy.io import fits
from astropy.utils.data import download_file
from astropy import units as u

from .. import registry
from .. import Spectrum, MagSystem, SpectralMagSystem, ABMagSystem

# ---------------------------------------------------------------------------
# AB system

def load_ab(name=None):
    return ABMagSystem(name=name)

registry.register_loader(MagSystem, 'ab', load_ab)


# ---------------------------------------------------------------------------
# Spectral systems

def load_spectral_magsys_fits(remote_url, name=None):
    fn = download_file(remote_url, cache=True)
    hdulist = fits.open(fn)
    dispersion = hdulist[1].data['WAVELENGTH']
    flux_density = hdulist[1].data['FLUX']
    hdulist.close()
    refspectrum = Spectrum(dispersion, flux_density, 
                           unit=(u.erg / u.s / u.cm**2 / u.AA), dunit=u.AA)
    return SpectralMagSystem(refspectrum, name=name)

calspec_url = 'ftp://ftp.stsci.edu/cdbs/current_calspec/'
vega_url = calspec_url + 'alpha_lyr_stis_005.fits'
registry.register_loader(MagSystem, 'vega', load_spectral_magsys_fits,
                         [vega_url], sourceurl=calspec_url)

bd17_url = calspec_url + 'bd_17d4708_stisnic_003.fits'
registry.register_loader(MagSystem, 'bd17', load_spectral_magsys_fits,
                         [bd17_url], sourceurl=calspec_url)

