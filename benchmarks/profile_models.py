import numpy as np
import sncosmo
import sncosmo.utils
from line_profiler import LineProfiler

model = sncosmo.get_model('hsiao')
band = sncosmo.get_bandpass('desg')

profiler = LineProfiler(model._flux.__call__)

def run():
    model.bandflux(np.array(100 * ['desg']), np.zeros(100, dtype=np.float))

#import cProfile
# .. was cProfile.run(..)
profiler.run('run()')
profiler.print_stats()
