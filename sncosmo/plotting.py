# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to plot light curve data and models."""
from __future__ import division

import math
import numpy as np

from .models import get_model
from .spectral import get_bandpass, get_magsystem
from .fitting import normalized_flux

__all__ = ['plotlc']

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
                       
def plotlc(data=None, model=None, fname=None, bands=None, show_pulls=True,
           include_model_error=False, xfigsize=None, yfigsize=None, dpi=100):
    """Plot light curve data or model light curves.

    Parameters
    ----------
    data : `~numpy.ndarray` or dict thereof, optional
        Structured array or dictionary of arrays, with the following fields:
        {'time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'}.
    model : `~sncosmo.Model`, optional
        If given, model light curve is plotted.
    fname : str, optional
        Filename to write plot to. If `None` (default), plot is shown using
        ``show()``.
    bands : list, optional
        List of Bandpasses, or names thereof, to plot.
    show_pulls : bool, optional
        If True (and if model and data are given), plot pulls. Default is
        ``True``.
    include_model_error : bool, optional
        Plot model error as a band around the model.
    xfigsize, yfigsize : float, optional
        figure size in inches in x or y. Specify one or the other, not both.
        Default is xfigsize=8.
    dpi : float, optional
        dpi to pass to ``plt.savefig()`` for rasterized images. 
        
    Examples
    --------

    Suppose we have data in a file that looks like this::
 
        time band flux fluxerr zp zpsys
        55070.0 sdssg -0.263064256628 0.651728140824 25.0 ab
        55072.0512821 sdssr -0.836688186816 0.651728140824 25.0 ab
        55074.1025641 sdssi -0.0104080573938 0.651728140824 25.0 ab
        55076.1538462 sdssz -0.0794771107707 0.651728140824 25.0 ab
        55078.2051282 sdssg 0.897840283912 0.651728140824 25.0 ab
        ...

    To read and plot the data:

        >>> meta, data = sncosmo.readlc('mydatafile.dat')  # read the data
        >>> sncosmo.plotlc(data, fname='plotlc_example.png')  # plot the data

    We happen to know the model and parameters that fit this
    data. Specifying the ``model`` keyword will plot the model over
    the data.
    
        >>> model = sncosmo.get_model('salt2')
        >>> model.set(z=0.5, c=0.2, t0=55100., mabs=-19.5, x1=0.5)
        >>> sncosmo.plotlc(data, fname='plotlc_example.png', model=model)

    .. image:: /pyplots/plotlc_example.png

    Plot just the model:

        >>> sncosmo.plotlc(model=model, bands=['sdssg', 'sdssr', 'sdssi', 'sdssz'], fname='plotlc_example.png')


    """

    from matplotlib import pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cmap = cm.get_cmap('gist_rainbow')
    cm_disp_range = (3000., 10000.)  # wavelengths corresponding to (blue, red)

    if data is None and model is None:
        raise ValueError('must specify at least one of: data, model')
    if data is None and bands is None:
        raise ValueError('must specify bands to plot for model')
    if model is not None:
        model = get_model(model)

    if data is not None:
        dataflux, datafluxerr = normalized_flux(data, zp=25., magsys='ab')

    # Bands to plot
    if data is None:
        bands = set([get_bandpass(band) for band in bands])
    else:
        data_bands = np.array([get_bandpass(band) for band in data['band']])
        unique_data_bands = set(data_bands)
        if bands is None:
            bands = unique_data_bands
        else:
            bands = set([get_bandpass(band) for band in bands])
            bands = bands & unique_data_bands
    bands = list(bands)
    disps = [b.disp_eff for b in bands]

    # Calculate layout of figure (columns, rows, figure size)
    nsubplots = len(bands)
    ncol = 2
    nrow = (nsubplots - 1) // ncol + 1
    if xfigsize is None and yfigsize is None:
        figsize = (4. * ncol, 3. * nrow)
    elif yfigsize is None:
        figsize = (xfigsize, 3. / 4. * nrow / ncol * xfigsize)
    elif xfigsize is None:
        figsize = (4. / 3. * ncol / nrow * yfigsize, yfigsize)
    else:
        raise ValueError('cannot specify both xfigsize and yfigsize')
    fig = plt.figure(figsize=figsize)

    axnum = 0
    for disp, band in sorted(zip(disps, bands)):
        axnum += 1

        color = cmap((cm_disp_range[1] - disp) /
                     (cm_disp_range[1] - cm_disp_range[0]))

        ax = plt.subplot(nrow, ncol, axnum)
        plt.text(0.9, 0.9, band.name, color='k', ha='right', va='top',
                 transform=ax.transAxes)
        if axnum % 2:
            plt.ylabel('flux ($ZP_{AB} = 25$)')

        xlabel_text = 'time'
        if model is not None and model.params['t0'] != 0.:
            xlabel_text += ' - {:.2f}'.format(model.params['t0'])

        if data is not None:
            idx = data_bands == band
            time = data['time'][idx]
            flux = dataflux[idx]
            fluxerr = datafluxerr[idx]

            if model is None:
                plotted_time = time
            else:
                plotted_time = time - model.params['t0']

            plt.errorbar(plotted_time, flux, fluxerr, ls='None',
                         color=color, marker='.', markersize=3.)

        if model is not None and model.bandoverlap(band):

            plotted_time = model.times() - model.params['t0']

            if include_model_error:
                modelflux, modelfluxerr = \
                    model.bandflux(band, zp=25., zpsys='ab',
                                   include_error=True)
            else:
                modelflux = model.bandflux(band, zp=25., zpsys='ab',
                                           include_error=False)

            plt.plot(plotted_time, modelflux, ls='-', marker='None',
                     color=color)
            if include_model_error:
                plt.fill_between(plotted_time, modelflux - modelfluxerr,
                                 modelflux + modelfluxerr, color=color,
                                 alpha=0.2)
            # maximum plot range
            ymin, ymax = ax.get_ylim()
            maxmodelflux = modelflux.max()
            ymin = max(ymin, -0.2 * maxmodelflux)
            ymax = min(ymax, 2. * maxmodelflux)
            ax.set_ylim(ymin, ymax)


        # steal part of the axes and plot pulls
        if show_pulls and data is not None and model is not None:
            divider = make_axes_locatable(ax)
            axpulls = divider.append_axes("bottom", size=0.7, pad=0.1,
                                          sharex=ax)
            modelflux = model.bandflux(band, time, zp=25., zpsys='ab') 
            pulls = (flux - modelflux) / fluxerr
            plt.plot(time - model.params['t0'], pulls, marker='.',
                     markersize=5., color=color, ls='None')
            plt.axhline(y=0., color=color)
            plt.setp(ax.get_xticklabels(), visible=False)
            if axnum % 2:
                plt.ylabel('pull')

        # label the most recent Axes x-axis
        plt.xlabel(xlabel_text)

    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97,
                        wspace=0.2, hspace=0.2)
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, dpi=dpi)
        plt.clf()
