import copy
import numpy as np
from spectral import get_bandpass, get_magsystem

__all__ = ['plotlc']

def _normalize_zp(data, zp=25., magsys='ab'):
    """Return flux values normalized to a common zeropoint and magnitude
    system."""

    magsys = get_magsystem(magsys)
    result = copy.deepcopy(data)

    for i in range(len(data)):
        ms = get_magsystem(data['zpsys'][i])
        factor = (ms.zpbandflux(data['band'][i]) / magsys.zpbandflux(data['band'][i]) *
                  10.**(0.4 * (zp - data['zp'][i])))
        result['flux'][i] *= factor
        result['fluxerr'][i] *= factor

    return result

def _time_like_field(data):
    for fieldname in ['time', 'date', 'mjd']:
        if fieldname in data.dtype.names:
            return fieldname
    raise ValueError('No time-like field found in data')


def plotlc(data, fname, model=None):
    """Plot light curves.

    Parameters
    ----------
    data : 

    fname : str

    """

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cmap = cm.get_cmap('gist_rainbow')
    disprange = (3000., 10000.)

    fig = plt.figure(figsize=(8., 6.))

    data = _normalize_zp(data, zp=25., magsys='ab')
    timefield = _time_like_field(data)

    bandnames = np.unique(data['band']).tolist()
    bands = [get_bandpass(bandname) for bandname in bandnames]
    disps = [0.5 * (band.disp[0] + band.disp[-1]) for band in bands]

    axnum = 0
    for disp, band, bandname in sorted(zip(disps, bands, bandnames)):
        axnum += 1

        idx = data['band'] == bandname
        time = data[timefield][idx]
        flux = data['flux'][idx]
        fluxerr = data['fluxerr'][idx]

        color = cmap((disprange[1] - disp) / (disprange[1] - disprange[0]))

        ax = plt.subplot(2, 2, axnum)
        plt.text(0.9, 0.9, bandname, color='k', ha='right', va='top',
                 transform=ax.transAxes)
        if axnum % 2:
            plt.ylabel('flux ($ZP_{AB} = 25$)')

        if model is None:
            plt.errorbar(time, flux, fluxerr, ls='None',
                         color=color, marker='.', markersize=3.)

        if model is not None:
            t0 = model.params['t0']
            plt.errorbar(time - t0, flux, fluxerr, ls='None',
                         color=color, marker='.', markersize=3.)

            modelflux = model.bandflux(band, zp=25., zpmagsys='ab')
            plt.plot(model.times() - t0, modelflux, ls='-', marker='None',
                     color=color)

            # steal part of the axes and plot pulls
            divider = make_axes_locatable(ax)
            axpulls = divider.append_axes("bottom", size=0.7, pad=0.1,
                                          sharex=ax)
            modelflux = model.bandflux(band, time, zp=25., zpmagsys='ab') 
            pulls = (flux - modelflux) / fluxerr
            plt.plot(time - t0, pulls, marker='.', markersize=5., color=color,
                     ls='None')
            plt.axhline(y=0., color=color)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.xlabel('time - {:.2f}'.format(t0))
            if axnum % 2:
                plt.ylabel('pull')

            # maximum plot range
            ymin, ymax = ax.get_ylim()
            maxmodelflux = modelflux.max()
            ymin = max(ymin, -0.2 * maxmodelflux)
            ymax = min(ymax, 2. * maxmodelflux)
            ax.set_ylim(ymin, ymax)

    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97,
                        wspace=0.2, hspace=0.2)
    plt.savefig(fname)
    plt.clf()
