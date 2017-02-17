"""Helper function to plot a set of bandpasses in sphinx docs."""
from __future__ import division

import numpy as np
from matplotlib import rc
from matplotlib import pyplot as plt

import sncosmo


def plot_bandpass_set(setname):
    """Plot the given set of bandpasses."""

    rc("font", family="serif")

    bandpass_meta = sncosmo.bandpasses._BANDPASSES.get_loaders_metadata()

    fig = plt.figure(figsize=(9, 3))
    ax = plt.axes()

    nbands = 0
    for m in bandpass_meta:
        if m['filterset'] != setname:
            continue
        print(m['name'])
        b = sncosmo.get_bandpass(m['name'])

        # add zeros on either side of bandpass transmission
        wave = np.zeros(len(b.wave) + 2)
        wave[0] = b.wave[0]
        wave[1:-1] = b.wave
        wave[-1] = b.wave[-1]
        trans = np.zeros(len(b.trans) + 2)
        trans[1:-1] = b.trans

        ax.plot(wave, trans, label=m['name'])
        nbands += 1

    ax.set_xlabel("Wavelength ($\\AA$)")
    ax.set_ylabel("Transmission")

    ncol = 1 + (nbands-1) // 9  # 9 labels per column
    ax.legend(loc='upper right', frameon=False, fontsize='small',
              ncol=ncol)

    # Looks like each legend column takes up about 0.125 of the figure.
    # Make room for the legend.
    xmin, xmax = ax.get_xlim()
    xmax += ncol * 0.125 * (xmax - xmin)
    ax.set_xlim(xmin, xmax)
    plt.tight_layout()
    plt.show()
