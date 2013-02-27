# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Reader functions for initializing built-in data."""

import tarfile
from os.path import join

from astropy.io import ascii
from astropy import units as u
from astropy.utils.data import (download_file, get_pkg_data_filename,
                                get_readable_fileobj)
from astropy.utils import OrderedDict
from astropy.config import ConfigurationItem

from . import registry
from . import Bandpass
from . import Model, TimeSeriesModel, SALT2Model
from . import utils

# -------------------------------------------------------------------------
# Bandpasses

def load_bandpass_ascii(pkg_data_name):
    """Read two-column bandpass. First column is assumed to be wavelength
    in Angstroms."""
    
    filename = get_pkg_data_filename(pkg_data_name)
    t = ascii.read(filename, names=['disp', 'trans'])
    return Bandpass(t['disp'], t['trans'], dispersion_unit=u.AA)

registry.register_loader(Bandpass, 'desg', load_bandpass_ascii,
                         ['data/bandpasses/des_g.dat'], filterset='des')
registry.register_loader(Bandpass, 'desr', load_bandpass_ascii,
                         ['data/bandpasses/des_r.dat'], filterset='des')
registry.register_loader(Bandpass, 'desi', load_bandpass_ascii,
                         ['data/bandpasses/des_i.dat'], filterset='des')
registry.register_loader(Bandpass, 'desz', load_bandpass_ascii,
                         ['data/bandpasses/des_z.dat'], filterset='des')
registry.register_loader(Bandpass, 'desy', load_bandpass_ascii,
                         ['data/bandpasses/des_y.dat'], filterset='des')


lines = [
    'The following bandpasses are available:',
    '========== ========================= ==============================',
    '   Name     Description               Source',
    '========== ========================= ==============================']
for d in registry.get_loaders_metadata(Bandpass):
    lines.append('{0:^10} {1:^25} {1:^25}'.format(d['name'], '', ''))
lines.append(lines[1])
bandpass_table = '\n'.join(lines)


# --------------------------------------------------------------------------
# Models

def load_timeseries_ascii(remote_url):
    with get_readable_fileobj(remote_url, cache=True) as f:
        phases, wavelengths, flux = utils.read_griddata(f)
        return TimeSeriesModel(phases, wavelengths, flux)

def load_salt2model(remote_url, topdir):
    fn = download_file(remote_url, cache=True)
    t = tarfile.open(fn, 'r:gz')

    errscalefn = join(topdir, 'salt2_spec_dispersion_scaling.dat')
    if errscalefn in t.getnames():
        errscalefile = t.extractfile(errscalefn)
    else:
        errscalefile = None

    m = SALT2Model(
        m0file=t.extractfile(join(topdir,'salt2_template_0.dat')),
        m1file=t.extractfile(join(topdir,'salt2_template_1.dat')),
        v00file=t.extractfile(join(topdir,'salt2_spec_variance_0.dat')),
        v11file=t.extractfile(join(topdir,'salt2_spec_variance_1.dat')),
        v01file=t.extractfile(join(topdir,'salt2_spec_covariance_01.dat')),
        errscalefile=errscalefile)
    t.close()
    return m


registry.register_loader(
    Model, 'nugent-sn1a', load_timeseries_ascii, 
    ['http://supernova.lbl.gov/~nugent/templates/sn1a_flux.v1.2.dat.gz'],
    version='1.2',
    sourceurl='http://supernova.lbl.gov/~nugent/templates')


baseurl = 'http://supernovae.in2p3.fr/~guy/salt/download/'
registry.register_loader(
    Model, 'salt2', load_salt2model,
    [baseurl + 'salt2_model_data-1-1.tar.gz', 'salt2-1-1'],
    version='1.1')
registry.register_loader(
    Model, 'salt2', load_salt2model,
    [baseurl + 'salt2_model_data-2-0.tar.gz', 'salt2-2-0'],
    version='2.0')
