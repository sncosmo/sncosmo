#!/usr/bin/env python
"""Make a 500x500 pixel png of the Hsiao spectrum at phase = 0"""

from matplotlib import cm
import matplotlib.pyplot as plt
import sncosmo

cmap = cm.get_cmap('gist_rainbow')
model = sncosmo.get_source('hsiao')

wave = model._wave
flux = model.flux(0., wave)

wmin = 2600.
wmax = 7400.
mask = (wave > wmin) & (wave < wmax)
wave = wave[mask]
flux = flux[mask]

cscale = (wmax - wave[1:]) / (wmax - wmin)
colors = cmap(cscale)

fig = plt.figure(figsize=(1., 1.), facecolor=(0., 0., 0., 0.))
ax = plt.axes([0., 0., 1., 1.], axisbg=(0.,0., 0., 0.))

for i in range(len(cscale)):
    d0, d1 = wave[i], wave[i+1]
    f0, f1 = flux[i], flux[i+1]
    plt.fill([d0, d0, d1, d1], [0., f0, f1, 0.], color=colors[i])
plt.xlim(wmin, wmax)
plt.ylim(0., 1.05*max(flux))
plt.axis('off')
plt.savefig('spectral.png', dpi=500., transparent=True)
plt.savefig('spectral_white_bkg.png', dpi=500., transparent=False)
