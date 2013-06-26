import numpy as np
import sncosmo

meta, data = sncosmo.readlc('simulated_salt2_data.dat')

model = sncosmo.get_model('salt2')
model.set(**meta)

sncosmo.plotlc(data, fname='plotlc_example.png', model=model)
