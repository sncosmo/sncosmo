"""Helper function for documentation to plot a set of bandpasses."""

from sncosmo import registry, Bandpass, get_bandpass
import matplotlib.pyplot as plt


def plot_bandpass_set(setname):
    """Plot the given set of bandpasses."""

    bandpass_meta = registry.get_loaders_metadata(Bandpass)

    fig = plt.figure(figsize=(9, 3))
    ax = plt.axes()

    for m in bandpass_meta:
        if m['filterset'] != setname:
            continue
        b = get_bandpass(m['name'])
        ax.plot(b.wave, b.trans, label=m['name'])

    ax.set_xlabel("Wavelength ($\\AA$)")
    ax.set_ylabel("Transmission")
    ax.legend(loc='upper right')

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(right=(xmax + 1000.))  # make room for legend
    plt.tight_layout()
    plt.show()
