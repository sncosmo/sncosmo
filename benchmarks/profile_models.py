import numpy as np
import sncosmo
from line_profiler import LineProfiler

model = sncosmo.get_model('hsiao')
band = sncosmo.get_bandpass('desg')

profiler = LineProfiler(model.bandflux)

bands = np.array(25*['desg'] + 25*['desr'] + 25*['desi'] + 25*['desz'])
times = np.arange(100) *0.5
zps = 31.4 * np.ones(100)
zpmagsys = np.array(100 * ['ab'])
def run():
    model.bandflux(bands, times, zp=zps, zpmagsys=zpmagsys)
run()
#import cProfile
# .. was cProfile.run(..)
profiler.run('run()')
profiler.print_stats()
