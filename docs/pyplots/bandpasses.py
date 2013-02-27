
import matplotlib.pyplot as plt
from sncosmo import registry
from sncosmo import Bandpass


bandpass_meta = registry.get_loaders_metadata(Bandpass)
filtersets = []
for m in bandpass_meta:
    if m['filterset'] not in filtersets: filtersets.append(m['filterset'])

fig = plt.figure(figsize=(6., 2. * len(filtersets)))
for i, filterset in enumerate(filtersets):
    for m in bandpass_meta:
        if m['filterset'] != filterset: continue
        b = Bandpass.from_name(m['name'])
        plt.plot(b.dispersion, b.transmission, label=m['name'])

plt.show()
