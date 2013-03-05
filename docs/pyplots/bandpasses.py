
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid
from sncosmo import registry
from sncosmo import Bandpass


bandpass_meta = registry.get_loaders_metadata(Bandpass)
filtersets = []
for m in bandpass_meta:
    if m['filterset'] not in filtersets: filtersets.append(m['filterset'])

fig = plt.figure(figsize=(9., 3. * len(filtersets)))
grid = Grid(fig, rect=111, nrows_ncols=(len(filtersets), 1),
            axes_pad=0.25, label_mode='L')

for ax, filterset in zip(grid, filtersets):

    for m in bandpass_meta:
        if m['filterset'] != filterset: continue
        b = Bandpass.from_name(m['name'])
        ax.plot(b.dispersion, b.transmission, label=m['name'])
    ax.set_xlabel('Angstroms')
    ax.set_ylabel('Transmission')
    ax.legend(loc='upper right')

xmin, xmax = ax.get_xlim()
ax.set_xlim(right=(xmax + 1000.))  # make room for legend
plt.tight_layout()
plt.show()
