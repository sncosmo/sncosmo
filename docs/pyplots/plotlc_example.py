import numpy as np
import sncosmo

data = sncosmo.readlc('simulated_salt2_data.dat')

model = sncosmo.ObsModel(source='salt2')
model.set(**data.meta)

sncosmo.plotlc(data, fname='plotlc_example.png', model=model)
