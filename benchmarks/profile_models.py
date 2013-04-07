import sncosmo

model = sncosmo.get_model('hsiao')
band = sncosmo.get_bandpass('desg')
def run():
    model.bandflux(band, 0.)

import cProfile
cProfile.run('run()', 'bandflux.pfl')
