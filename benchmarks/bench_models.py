"""Run benchmarks for model synthetic photometry"""

import time
import os
import glob
import argparse
from collections import OrderedDict
import numpy as np
import sncosmo

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-l", "--label", dest="label", default=None, 
                    help="Save results to a pickle with this label, so that"
                    "they can be displayed later. Pickles are save in a "
                    "'_results' directory in the same parent directory as "
                    "this script.")
parser.add_argument("-s", "--show", dest="show", action="store_true",
                    default=False, help="Show all results from previously "
                    "labeled runs")
parser.add_argument("-d", "--delete", dest="delete_label", default=None,
                    help="Delete saved results with the given label "
                    "(or 'all' to delete all results). Do not run any "
                    "benchmarks.")
args = parser.parse_args()

if args.show and args.label:
    parser.error("--label doesn't do anything when --show is specified.")
if args.delete_label and (args.label or args.show):
    parser.error("--label and --show do not do anything when --delete is "
                 "specified.")

resultsdir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          '_results', 'flux'))
piknames = glob.glob(os.path.join(resultsdir, '*.pik'))

# Delete saved results, if requested.
if args.delete_label is not None:
    if args.delete_label.lower() == 'all':
        for pikname in piknames:
            os.remove(pikname)
    else:
        try:
            os.remove(os.path.join(resultsdir,
                                   '{0}.pik'.format(args.delete_label)))
        except OSError:
            raise ValueError('No such label exists: {0}'
                             .format(args.delete_label))
    exit()

c = OrderedDict()

# test data
ndata = 100
dates = np.ones(ndata, dtype=np.float)
bands = np.array(ndata * ['desr'])
niter = 50

# models
salt2 = sncosmo.get_model('salt2')
hsiao = sncosmo.get_model('hsiao')
salt2_set = salt2(z=0.5, c=0., x1=0., absmag=(-19.3, 'bessellb', 'ab'))
hsiao_set = hsiao(z=0.5, c=0., absmag=(-19.3, 'bessellb', 'ab'))
models = [salt2, hsiao, salt2_set, hsiao_set]
modelnames = ['salt2', 'hsiao', 'salt2 z=0.5', 'hsiao z=0.5']

print "\nmodel.flux()"
print 60 * "-"
print "Model      1          10         100 dates   millisec per call"
print 60 * "-"
for modelname, model in zip(modelnames, models):
    print '{:15s}'.format(modelname),
    for idx in [0, range(10), range(100)]:
        d = dates[idx]
        time1 = time.time()
        for i in range(niter): model.flux(d)
        time2 = time.time()
        time_sec = (time2 - time1) / niter
        print "%10.5f" % (time_sec * 1000.),
    print ""

print "\nmodel.bandflux()"
print 60 * "-"
print "Model      1          10         100 dates   millisec per call"
print 60 * "-"
for modelname, model in zip(modelnames, models):
    print '{:15s}'.format(modelname),
    for idx in [0, range(10), range(100)]:
        d = dates[idx]
        b = bands[idx]
        time1 = time.time()
        for i in range(niter): model.bandflux(b, d)
        time2 = time.time()
        time_sec = (time2 - time1) / niter
        print "%10.5f" % (time_sec * 1000.),
    print ""
