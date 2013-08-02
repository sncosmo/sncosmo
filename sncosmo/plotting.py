# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to plot light curve data and models."""
from __future__ import division

import math
import numpy as np

from astropy.utils.misc import isiterable

from .models import get_model
from .spectral import get_bandpass, get_magsystem
from .photometric_data import PhotData

__all__ = ['plotlc', 'animate_model']

# TODO: cleanup names: data_bands, etc 
# TODO: standardize docs for `data` in this and other functions.
# TODO: better example(s)
# TODO: return the Figure?
def plotlc(data=None, model=None, bands=None, show_pulls=True,
           include_model_error=False, zp=25., zpsys='ab',
           xfigsize=None, yfigsize=None, dpi=100, fname=None):
    """Plot light curve data or model light curves.

    Parameters
    ----------
    data : `~numpy.ndarray` or dict thereof, optional
        Structured array or dictionary of arrays, with certain required fields.
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
    zp : float, optional
        Zeropoint to normalize the flux. Default is 25.
    zpsys : str or `~sncosmo.MagSystem`, optional
        Zeropoint system for `zp`. Default is 'ab'.
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
        >>> sncosmo.plotlc(data, model=model, fname='plotlc_example.png',)

    .. image:: /pyplots/plotlc_example.png

    Plot just the model:

        >>> sncosmo.plotlc(model=model, bands=['sdssg', 'sdssr', 'sdssi', 'sdssz'], fname='plotlc_example.png')


    """

    from matplotlib import pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Get colormap and define wavelengths corresponding to (blue, red)
    cmap = cm.get_cmap('gist_rainbow')
    cm_disp_range = (3000., 10000.)

    if data is None and model is None:
        raise ValueError('must specify at least one of: data, model')
    if data is None and bands is None:
        raise ValueError('must specify bands to plot for model')
    if model is not None:
        model = get_model(model)

    if data is not None:
        data = PhotData(data)
        nflux, nfluxerr = data.normalized_flux(zp=zp, zpsys=zpsys,
                                               include_err=True)

    # Bands to plot
    if data is None:
        bands = set([get_bandpass(band) for band in bands])
    else:
        data_bands = np.array([get_bandpass(band) for band in data.band])
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
            plt.ylabel('flux ($ZP_{' + get_magsystem(zpsys).name.upper() +
                       '} = ' + str(zp) + '$)')

        xlabel_text = 'time'
        if model is not None and model.params['t0'] != 0.:
            xlabel_text += ' - {:.2f}'.format(model.params['t0'])

        if data is not None:
            idx = data_bands == band
            time = data.time[idx]
            flux = nflux[idx]
            fluxerr = nfluxerr[idx]

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
                    model.bandflux(band, zp=zp, zpsys=zpsys,
                                   include_error=True)
            else:
                modelflux = model.bandflux(band, zp=zp, zpsys=zpsys,
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
            modelflux = model.bandflux(band, time, zp=zp, zpsys=zpsys) 
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


def animate_model(model_or_models, fps=30, length=20.,
                  time_range=(None, None), disp_range=(None, None),
                  match_refphase=True, match_flux=True, fname=None):
    """Animate a model's SED using matplotlib.animation. (requires
    matplotlib v1.1 or higher).

    Parameters
    ----------
    model_or_models : `~sncosmo.Model` or str or iterable thereof
        The model to animate or list of models to animate.
    fps : int, optional
        Frames per second. Default is 30.
    length : float, optional
        Movie length in seconds. Default is 15.
    time_range : (float, float), optional
        Time range to plot (in the timeframe of the first model if multiple
        models are given). `None` indicates to use the maximum extent of the
        model(s).
    disp_range : (float, float), optional
        Dispersion range to plot. `None` indicates to use the maximum extent
        of the model(s).
    match_flux : bool, optional
        For multiple models, scale fluxes so that the peak of the spectrum
        at the reference phase matches that of the first model. Default is
        False.
    match_refphase : bool, optional
        For multiple models, shift additional models so that the model's
        reference phase matches that of the first model.
    fname : str, optional
        If not `None`, save animation to file `fname`. Requires ffmpeg
        to be installed with the appropriate codecs: If `fname` has
        the extension '.mp4' the libx264 codec is used. If the
        extension is '.webm' the VP8 codec is used. Otherwise, the
        'mpeg4' codec is used. The first frame is also written to a
        png.

    Examples
    --------

    Compare the salt2 and hsiao models::

        animate_model(['salt2', 'hsiao'],  time_range=(None, 30.),
                      disp_range=(2000., 9200.))

    Compare the salt2 model with x1 = 1. to the same model with x1 = 0.::

        m1 = sncosmo.get_model('salt2')
        m1.set(x1=1.)
        m2 = sncosmo.get_model('salt2')
        m2.set(x1=0.)
        animate_model([m1, m2])

    """

    from matplotlib import pyplot as plt
    from matplotlib import animation

    # get the model(s)
    if (not isiterable(model_or_models) or
        isinstance(model_or_models, basestring)):
        model_or_models = [model_or_models]
    models = [get_model(m) for m in model_or_models]

    # time offsets needed to match refphases
    time_offsets = [model.refphase - models[0].refphase for model in models]
    if not match_refphase:
        time_offsets = [0.] * len(models)

    # determine times to display
    model_min_times = [models[i].times()[0] - time_offsets[i] for
                       i in range(len(models))]
    model_max_times = [models[i].times()[-1] - time_offsets[i] for
                       i in range(len(models))]
    min_time, max_time = time_range
    if min_time is None:
        min_time = min(model_min_times)
    if max_time is None:
        max_time = max(model_max_times)

    
    # determine the min and max dispersions
    disps = [model.disp() for model in models]
    min_disp, max_disp = disp_range
    if min_disp is None:
        min_disp = min([d[0] for d in disps])
    if max_disp is None:
        max_disp = max([d[-1] for d in disps])

    # model time interval between frames
    time_interval = (max_time - min_time) / (length * fps)

    # maximum flux density of each model at the refphase
    max_fluxes = [np.max(model.flux(model.refphase)) for model in models]

    # scaling factors
    if match_flux:
        max_bandfluxes = [model.bandflux(model.refband, model.refphase)
                          for model in models]
        scaling_factors = [max_bandfluxes[0] / f for f in max_bandfluxes]
        global_max_flux = max_fluxes[0]
    else:
        scaling_factors = [1.] * len(models)
        global_max_flux = max(max_fluxes)

    ymin = -0.06 * global_max_flux
    ymax = 1.1 * global_max_flux

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(min_disp, max_disp), ylim=(ymin, ymax))
    plt.axhline(y=0., c='k')
    plt.xlabel('Wavelength ($\\AA$)') 
    plt.ylabel('Flux Density ($F_\lambda$)')
    time_text = ax.text(0.05, 0.95, '', ha='left', va='top',
                        transform=ax.transAxes)
    empty_lists = 2 * len(models) * [[]]
    lines = ax.plot(*empty_lists, lw=1)
    for line, model in zip(lines, models):
        line.set_label(model.name)
    legend = plt.legend(loc='upper right')

    def init():
        for line in lines:
            line.set_data([], [])
        time_text.set_text('')
        return tuple(lines) + (time_text,)

    def animate(i):
        current_time = min_time + time_interval * i
        for j in range(len(models)):
            y = models[j].flux(current_time + time_offsets[j])
            lines[j].set_data(disps[j], y * scaling_factors[j])
        time_text.set_text('time (days) = {:.1f}'.format(current_time))
        return tuple(lines) + (time_text,)

    # Call the animator.
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=int(fps*length), interval=(1000./fps),
                                  blit=True)

    # Save the animation as an mp4. This requires that ffmpeg is installed.
    if fname is not None:
        i = fname.rfind('.')
        stillfname = fname[:i] + '.png'
        plt.savefig(stillfname) 

        ext = fname[i+1:]
        codec = {'mp4': 'libx264', 'webm': 'libvpx'}.get(ext, 'mpeg4')
        ani.save(fname, fps=fps, codec=codec, extra_args=['-vcodec', codec],
                 writer='ffmpeg_file', bitrate=1800)

    else:
        plt.show()
