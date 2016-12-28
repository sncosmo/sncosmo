#!/usr/bin/env python
import os
import numpy as np
import sncosmo

# generate arrays to input to interpolator
x = np.array([-1., 0., 2., 4., 5., 6., 6.5, 7.])
y = np.array([1., 2., 3., 4., 5.])
z = np.sin(x)[:, None] * np.cos(0.25 * y)

fname = '../sncosmo/tests/data/interpolation_test_input.dat'
if os.path.exists(fname):
    print(fname, "already exists; skipping.")
else:
    sncosmo.write_griddata_ascii(x, y, z, fname)
    print("wrote", fname)


# generate test x and y arrays
xs = np.array([-2., -1., 0.5, 2.4, 3.0, 4.0, 4.5, 6.5, 8.0])
ys = np.array([0., 0.5, 1.0, 1.5, 2.8, 3.5, 4.0, 4.5, 5.0, 6.0])
for arr, name in ((xs, 'x'), (ys, 'y')):
    fname = '../sncosmo/tests/data/interpolation_test_eval{}.dat'.format(name)
    if os.path.exists(fname):
        print(fname, "already exists; skipping.")
    else:
        np.savetxt(fname, arr)
