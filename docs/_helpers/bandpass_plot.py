"""Helper function to plot a set of bandpasses in sphinx docs."""
from __future__ import division

import sncosmo
from matplotlib import rc
from matplotlib import pyplot as plt


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
        b = sncosmo.get_bandpass(m['name'])
        ax.plot(b.wave, b.trans, label=m['name'])
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
