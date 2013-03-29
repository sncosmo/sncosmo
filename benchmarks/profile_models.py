import sncosmo

sncosmo.get_model('hsiao')
def run():
    model.bandflux('desg', 0.)

import cProfile
cProfile.run('run()', 'bandflux.profile')
