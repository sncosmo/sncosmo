#!/usr/bin/env python
import numpy as np
import sncosmo
from collections import OrderedDict as odict
from astropy.table import Table


model = sncosmo.Model(source='salt2')
model.set(z=0.5, c=0.2, t0=55100., x1=0.5)
model.set_source_peakabsmag(-19.5, 'bessellb', 'ab')

times = np.linspace(55070., 55150., 40)
bands = np.array(10 * ['sdssg', 'sdssr', 'sdssi', 'sdssz'])
zp = 25. * np.ones(40)
zpsys = np.array(40 * ['ab'])

flux = model.bandflux(bands, times, zp=zp, zpsys=zpsys)
fluxerr = (0.05 * np.max(flux)) * np.ones(40, dtype=np.float)
flux += fluxerr * np.random.randn(40)

data = Table(odict([('time', times), ('band', bands), ('flux', flux),
                    ('fluxerr', fluxerr), ('zp', zp), ('zpsys', zpsys)]),
             meta=dict(zip(model.param_names, model.parameters)))

sncosmo.write_lc(data, 'example_photometric_data.dat')
