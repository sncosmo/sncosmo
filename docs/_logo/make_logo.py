#!/usr/bin/env python

from matplotlib import cm
import matplotlib.pyplot as plt
import sncosmo
cmap = cm.get_cmap('gist_rainbow')
model = sncosmo.get_model('hsiao')

disp = model.disp()
flux = model.flux(0.)

dmin = 2600.
dmax = 7400.
idx = (disp > dmin) & (disp < dmax)
disp = disp[idx]
flux = flux[idx]

cscale = (dmax - disp[1:]) / (dmax - dmin)
colors = cmap(cscale)

fig = plt.figure(figsize=(1., 1.), facecolor=(0., 0., 0., 0.))
ax = plt.axes([0., 0., 1., 1.], axisbg=(0.,0., 0., 0.))

for i in range(len(cscale)):
    d0, d1 = disp[i], disp[i+1]
    f0, f1 = flux[i], flux[i+1]
    plt.fill([d0, d0, d1, d1], [0., f0, f1, 0.], color=colors[i])
plt.xlim(dmin, dmax)
plt.ylim(0., 1.05*max(flux))
plt.axis('off')
plt.savefig('spectral.png', dpi=500., transparent=True)
