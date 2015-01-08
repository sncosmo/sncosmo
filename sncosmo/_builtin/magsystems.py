# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Importing this module registers loaders for built-in magnitude systems.
# The module docstring, a table of the magnitude systems, is generated at the
# after all the magnitude systems are registered.

import string

from astropy.io import fits
from astropy.utils.data import download_file
from astropy import units as u
from astropy.extern import six

from .. import registry
from .. import Spectrum, MagSystem, SpectralMagSystem, ABMagSystem


# ---------------------------------------------------------------------------
# AB system
def load_ab(name=None):
    return ABMagSystem(name=name)
registry.register_loader(
    MagSystem, 'ab', load_ab,
    meta={'subclass': '`~sncosmo.ABMagSystem`',
          'description': 'Source of 3631 Jy has magnitude 0 in all bands'})


# ---------------------------------------------------------------------------
# Spectral systems
def load_spectral_magsys_fits(remote_url, name=None):
    fn = download_file(remote_url, cache=True)
    hdulist = fits.open(fn)
    dispersion = hdulist[1].data['WAVELENGTH']
    flux_density = hdulist[1].data['FLUX']
    hdulist.close()
    refspectrum = Spectrum(dispersion, flux_density,
                           unit=(u.erg / u.s / u.cm**2 / u.AA), wave_unit=u.AA)
    return SpectralMagSystem(refspectrum, name=name)

calspec_url = 'ftp://ftp.stsci.edu/cdbs/calspec/'

registry.register_loader(
    MagSystem, 'vega', load_spectral_magsys_fits,
    args=[calspec_url + 'alpha_lyr_stis_007.fits'],
    meta={'subclass': '`~sncosmo.SpectralMagSystem`', 'url': calspec_url,
          'description': 'Vega (alpha lyrae) has magnitude 0 in all bands'})

registry.register_loader(
    MagSystem, 'bd17', load_spectral_magsys_fits,
    args=[calspec_url + 'bd_17d4708_stisnic_005.fits'],
    meta={'subclass': '`~sncosmo.SpectralMagSystem`', 'url': calspec_url,
          'description': 'BD+17d4708 has magnitude 0 in all bands.'})

del calspec_url

# --------------------------------------------------------------------------
# Generate docstring

lines = ['',
         '  '.join([10*'=', 60*'=', 35*'=', 15*'=']),
         '{0:10}  {1:60}  {2:35}  {3:15}'
         .format('Name', 'Description', 'Subclass', 'Spectrum Source')]
lines.append(lines[1])

urlnums = {}
for m in registry.get_loaders_metadata(MagSystem):

    urllink = ''
    description = ''

    if 'description' in m:
        description = m['description']

    if 'url' in m:
        url = m['url']
        if url not in urlnums:
            if len(urlnums) == 0:
                urlnums[url] = 0
            else:
                urlnums[url] = max(urlnums.values()) + 1
        urllink = '`{0}`_'.format(string.ascii_letters[urlnums[url]])

    lines.append("{0!r:10}  {1:60}  {2:35}  {3:15}"
                 .format(m['name'], description, m['subclass'], urllink))

lines.extend([lines[1], ''])
for url, urlnum in six.iteritems(urlnums):
    lines.append('.. _`{0}`: {1}'.format(string.ascii_letters[urlnum], url))
lines.append('')
__doc__ = '\n'.join(lines)

del lines
del urlnums
