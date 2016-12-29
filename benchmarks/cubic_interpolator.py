#!/usr/bin/env python
"""
Quick script to show difference between RectBivariateSpline and
the thing used in snfit.
"""
import time

import numpy as np
from scipy.interpolate import RectBivariateSpline
from sncosmo.interpolation import BicubicInterpolator
import matplotlib.pyplot as plt

def f_true(x, y):
    return np.sin(x)[:, None] * np.cos(0.25 * y)

def compare():
    x = np.array([-1., 0., 2., 4., 5., 6., 6.5, 7.])
    y = np.array([1., 2., 3., 4., 5.])
    z = f_true(x, y)

    f = BicubicInterpolator(x, y, z)

    xs = np.linspace(x[0], x[-1], 100)
    ys = np.linspace(y[0], y[-1], 100)
    zs_true = f_true(xs, ys)
    zs = f(xs, ys)
    zs_diff = zs - zs_true
    
    # compare to BivariateSpline
    f2 = RectBivariateSpline(x, y, z, kx=3, ky=3)
    zs2 = f2(xs, ys)
    zs2_diff = zs2 - zs_true

    # set min,max for diffs
    vmin = zs_diff.min()
    vmax = zs_diff.max()
    
    f, ax = plt.subplots(nrows=3, ncols=2, figsize=(10., 10.))
    ax[0,0].imshow(zs_true)
    ax[0,0].set_title('True function')
    ax[0,1].set_visible(False)
    
    ax[1,0].imshow(zs)
    ax[2,0].imshow(zs_diff, vmin=vmin, vmax=vmax)
    ax[1,0].set_title('BicubicInterpolator (snfit)')
    ax[2,0].set_title('difference')
    
    ax[1,1].imshow(zs2)
    ax[2,1].imshow(zs2_diff, vmin=vmin, vmax=vmax)
    ax[1,1].set_title('RectBivariateSpline (scipy)')
    ax[2,1].set_title('difference')

    f.savefig('cubic_interpolator_comparison.png')


def timeit():
    x = np.arange(100)
    y = np.arange(1000)
    z = np.sin(x)[:, None] * np.cos(0.25 * y)

    f1 = BicubicInterpolator(x, y, z)
    f2 = RectBivariateSpline(x, y, z, kx=3, ky=3)

    # Test data. In typical use, we'll have ~15 lightcurve points and
    # ~100 bandpass points (y)
    xs = np.linspace(10., 90., 15)
    ys = np.linspace(200., 400., 100)

    nloops = 10000
    for (name, f) in (('BicubicInterpolator', f1), ('RectBivariateSpline', f2)):
        t0 = time.time()
        for _ in range(nloops):
            f(xs, ys)
        t = time.time() - t0
        print('{} : {:.3f} us per evaluation'.format(name, t/nloops * 1e6))

timeit()    
