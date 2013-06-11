# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to plot light curve data and models."""

import numpy as np

from .spectral import get_bandpass, get_magsystem
from .fitting import normalized_flux

__all__ = ['plotlc', 'plotpdfs']

def normalized_flux(data, zp=25., magsys='ab'):
    """Return flux values normalized to a common zeropoint and magnitude
    system."""

    datalen = len(data['flux'])
    magsys = get_magsystem(magsys)
    flux = np.empty(datalen, dtype=np.float)
    fluxerr = np.empty(datalen, dtype=np.float)

    for i in range(datalen):
        ms = get_magsystem(data['zpsys'][i])
        factor = (ms.zpbandflux(data['band'][i]) /
                  magsys.zpbandflux(data['band'][i]) *
                  10.**(0.4 * (zp - data['zp'][i])))
        flux[i] = data['flux'][i] * factor
        fluxerr[i] = data['fluxerr'][i] * factor

    return flux, fluxerr
                       
def plotlc(data, fname=None, model=None, show_pulls=True,
           include_model_error=False):
    """Plot light curve.

    Parameters
    ----------
    data : `~numpy.ndarray` or dict thereof
        Structured array or dictionary of arrays, with the following fields:
        {'time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'}.
    fname : str, optional
        Filename to write plot to. If `None` (default), plot is shown using
        ``show()``.
    model : `~sncosmo.Model`, optional
        If given, model light curve is overplotted.
    show_pulls : bool, optional
        If True (and if model is given), plot pulls. Default is ``True``.
    """

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cmap = cm.get_cmap('gist_rainbow')
    disprange = (3000., 10000.)

    fig = plt.figure(figsize=(8., 6.))

    dataflux, datafluxerr = normalized_flux(data, zp=25., magsys='ab')

    bandnames = np.unique(data['band']).tolist()
    bands = [get_bandpass(bandname) for bandname in bandnames]
    disps = [b.disp_eff for b in bands]

    axnum = 0
    for disp, band, bandname in sorted(zip(disps, bands, bandnames)):
        axnum += 1

        idx = data['band'] == bandname
        time = data['time'][idx]
        flux = dataflux[idx]
        fluxerr = datafluxerr[idx]

        color = cmap((disprange[1] - disp) / (disprange[1] - disprange[0]))

        ax = plt.subplot(2, 2, axnum)
        plt.text(0.9, 0.9, bandname, color='k', ha='right', va='top',
                 transform=ax.transAxes)
        if axnum % 2:
            plt.ylabel('flux ($ZP_{AB} = 25$)')

        if model is None:
            plt.errorbar(time, flux, fluxerr, ls='None',
                         color=color, marker='.', markersize=3.)

        if model is not None and model.bandoverlap(band):
            t0 = model.params['t0']
            plt.errorbar(time - t0, flux, fluxerr, ls='None',
                         color=color, marker='.', markersize=3.)

            result = model.bandflux(band, zp=25., zpsys='ab',
                                    include_error=include_model_error)
            if include_model_error:
                modelflux, modelfluxerr = result
            else:
                modelflux = result

            plt.plot(model.times() - t0, modelflux, ls='-', marker='None',
                     color=color)
            if include_model_error:
                plt.fill_between(model.times() - t0, modelflux - modelfluxerr,
                                 modelflux + modelfluxerr, color=color,
                                 alpha=0.2)


            # steal part of the axes and plot pulls
            if show_pulls:
                divider = make_axes_locatable(ax)
                axpulls = divider.append_axes("bottom", size=0.7, pad=0.1,
                                              sharex=ax)
                modelflux = model.bandflux(band, time, zp=25., zpsys='ab') 
                pulls = (flux - modelflux) / fluxerr
                plt.plot(time - t0, pulls, marker='.', markersize=5.,
                         color=color, ls='None')
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
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
    plt.clf()

def plotpdfs(parnames, samples, weights, averages, errors, fname):
    """
    Plot PDFs of parameter values.
    
    Parameters
    ----------
    parnames : list of str
        Parameter names.
    samples : `~numpy.ndarray` (nsamples, nparams)
        Parameter values.
    weights : `~numpy.ndarray` (nsamples)
        Weight of each sample.
    fname : str
        Output filename.
    """
    import matplotlib.pyplot as plt

    npar = len(parnames)


    for i in range(max(4, npar)):

        plot_range = (averages[i] - 5*errors[i], averages[i] + 5*errors[i])
        plot_text = '{} = {:f} +/- {:f}'.format(parnames[i], averages[i],
                                                errors[i])

        ax = plt.subplot(2,2,i)
        plt.hist(samples[:, i], weights=weights, range=plot_range,
                 bins=30)
        plt.text(0.9, 0.9, plot_text, color='k', ha='right', va='top',
                 transform=ax.transAxes)

    plt.savefig(fname)
    plt.clf()
