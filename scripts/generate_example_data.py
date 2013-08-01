#!/usr/bin/env python
import numpy as np
import sncosmo
from astropy.utils import OrderedDict as odict

model = sncosmo.get_model('salt2')
params = odict(z=0.5, c=0.2, t0=55100., mabs=-19.5, x1=0.5)
model.set(**params)

times = np.linspace(55070., 55150., 40)
bands = np.array(10 * ['sdssg', 'sdssr', 'sdssi', 'sdssz'])
zp = 25. * np.ones(40)
zpsys = np.array(40 * ['ab'])

flux = model.bandflux(bands, times, zp=zp, zpsys=zpsys)
fluxerr = (0.05 * np.max(flux)) * np.ones(40, dtype=np.float)
flux += fluxerr * np.random.randn(40)

data = odict([('time', times), ('band', bands), ('flux', flux), 
              ('fluxerr', fluxerr), ('zp', zp), ('zpsys', zpsys)])

params['model'] = 'salt2'
sncosmo.writelc(data, 'example_photometric_data.dat', meta=params)
