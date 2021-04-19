#!/usr/bin/env python
import numpy as np
import sncosmo

wave = np.arange(3000, 9000, 10)
model = sncosmo.Model(source='hsiao-subsampled')
model.set(z=0.1, amplitude=1e-5, t0=0.)
sample_time = model['t0'] + 2.
flux = model.flux(time=sample_time, wave=wave)
fluxerr = 0.1 * flux
flux = flux + np.random.normal(scale=fluxerr)

np.savetxt('example_spectrum.dat', np.array([wave, flux, fluxerr]).T,
           header="wave flux fluxerr")
