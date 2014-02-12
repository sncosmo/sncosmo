"""Run benchmarks for model synthetic photometry"""

import time
import os
import glob
import argparse
from collections import OrderedDict
import numpy as np
import sncosmo

delim = 61 * "-"

# test data
ndata = 100 # make divisible by 4!
dates = np.linspace(-15., 40., ndata)
bands = np.array((ndata/4) * ['desg', 'desr', 'desi', 'sdssg'])
niter = 100

# models
f99dust = sncosmo.F99Dust(3.1)
models = OrderedDict([
    ('salt2', sncosmo.Model(source='salt2')),
    ('hsiao', sncosmo.Model(source='hsiao')),
    ('salt2+f99dust',
     sncosmo.Model(source='salt2', effects=[f99dust],
                   effect_names=['mw'], effect_frames=['obs'])),
    ('hsiao+f99dust',
     sncosmo.Model(source='hsiao', effects=[f99dust],
                   effect_names=['mw'], effect_frames=['obs']))
    ])

print "\nbandflux(band_array, time_array) [4 des bands]:"
print delim
print "Model              n=1        n=10       n=100"
print delim
for name, model in models.iteritems():
    print '{:15s}'.format(name),
    for idx in [0, range(10), range(100)]:
        d = dates[idx]
        b = bands[idx]
        time1 = time.time()
        for i in range(niter): model.bandflux(b, d)
        time2 = time.time()
        time_sec = (time2 - time1) / niter
        print "%10.5f" % (time_sec * 1000.),
    print " ms per call"
