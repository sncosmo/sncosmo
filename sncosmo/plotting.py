# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to plot light curve data and models."""
from __future__ import division

import math

import numpy as np
from astropy.utils import deprecated
from astropy.utils.misc import isiterable

from .models import SourceModel, ObsModel, get_sourcemodel
from .spectral import get_bandpass, get_magsystem
from .photdata import standardize_data, normalize_data
from .utils import value_error_str

__all__ = ['plot_lc', 'plot_param_samples', 'animate_model']

_model_ls = ['-', '--', ':', '-.']

# TODO: change plot_lc and animate_model() to return Figures like
# triangle.corner()?

# TODO: cleanup names: data_bands, etc 
# TODO: standardize docs for `data` in this and other functions.
def plot_lc(data=None, model=None, bands=None, zp=25., zpsys='ab', pulls=True,
            offsets=None, xfigsize=None, yfigsize=None, figtext=None,
            errors=None, figtextsize=1., fname=None, **kwargs):
    """Plot light curve data or model light curves.

    Parameters
    ----------
    data : `~numpy.ndarray` or dict of list_like, optional
        Structured array or dictionary of arrays or lists.
    model : `~sncosmo.ObsModel` or list thereof
        If given, model light curve is plotted. If a string, the corresponding
        model is fetched from the registry. If a list or tuple of
        `~sncosmo.ObsModel`, multiple models are plotted.
    bands : list, optional
        List of Bandpasses, or names thereof, to plot.
    zp : float, optional
        Zeropoint to normalize the flux. Default is 25.
    zpsys : str or `~sncosmo.MagSystem`, optional
        Zeropoint system for `zp`. Default is 'ab'.
    pulls : bool, optional
        If True (and if model and data are given), plot pulls. Default is
        ``True``.
    offsets : list, optional
        Offsets in flux for given bandpasses.
    figtext : str, optional
        Text to add to top of figure. If a list of strings, each item is
        placed in a separate "column". Use newline separators for multiple
        lines.
    xfigsize, yfigsize : float, optional
        figure size in inches in x or y. Specify one or the other, not both.
        Default is xfigsize=8.
    figtextsize : float, optional
        Space to reserve at top of figure for figtext (if not None).
        Default is 1 inch.
    fname : str, optional
        Filename to pass to savefig. If `None` (default), plot is shown
        using `~matplotlib.pyplot.show()`.
    kwargs :
        Any additional keyword args are passed to `~matplotlib.pyplot.savefig`.
        Popular options include ``dpi``, ``format``, ``transparent``. See
        matplotlib docs for full list.

    Examples
    --------
    Load some example data:

    >>> import sncosmo
    >>> data = sncosmo.load_example_data()

    Plot the data:

    >>> sncosmo.plot_lc(data)  # doctest: +SKIP

    Plot a model along with the data:
    
    >>> model = sncosmo.ObsModel('salt2')                # doctest: +SKIP
    >>> model.set(z=0.5, c=0.2, t0=55100., x0=1.547e-5)  # doctest: +SKIP
    >>> sncosmo.plot_lc(data, model=model)               # doctest: +SKIP

    .. image:: /pyplots/plotlc_example.png

    Plot just the model, for selected bands:

    >>> sncosmo.plot_lc(model=model,                     # doctest: +SKIP
    ...                 bands=['sdssg', 'sdssr'])        # doctest: +SKIP

    Plot figures on a multipage pdf:

    >>> from matplotlib.backends.backend_pdf import PdfPages  # doctest: +SKIP
    >>> pp = PdfPages('output.pdf')                           # doctest: +SKIP
    ...
    >>> # Do the following as many times as you like:
    >>> sncosmo.plot_lc(data, fname=pp, format='pdf')    # doctest: +SKIP
    ...
    >>> # Don't forget to close at the end:
    >>> pp.close()                                       # doctest: +SKIP

    """

    from matplotlib import pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Get colormap and define wavelengths corresponding to (blue, red)
    cmap = cm.get_cmap('gist_rainbow')
    cm_wave_range = (3000., 10000.)

    if data is None and model is None:
        raise ValueError('must specify at least one of: data, model')
    if data is None and bands is None:
        raise ValueError('must specify bands to plot for model(s)')

    # Get the model(s).
    if model is None:
        models = []
    elif isinstance(model, (tuple, list)):
        models = model
    else:
        models = [model]
    if not all([isinstance(m, ObsModel) for m in models]):
        raise TypeError('model(s) must be ObsModel instance(s)')

    # Standardize and normalize data.
    if data is not None:
        data = standardize_data(data)
        data = normalize_data(data, zp=zp, zpsys=zpsys)

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
    waves = [b.wave_eff for b in bands]

    # offsets for each band, if any.
    if offsets is not None:
        for key, value in offsets.iteritems():
            offsets[get_bandpass(key)] = offsets.pop(key)
        for band in bands:
            if band not in offsets:
                offsets[band] = 0.

    # Initialize errors
    if errors is None:
        errors = {}

    # Build figtext if not given explicitly
    if figtext is None:
        figtext = []
    elif isinstance(figtext, basestring):
        figtext = [figtext]
        
    if len(models) == 1:
        model = models[0]
        lines = []
        for i in range(len(model.param_names)):
            name = model.param_names[i]
            lname = model.param_names_latex[i]
            if name in errors:
                v = value_error_str(model.parameters[i], errors[name],
                                    latex=True)
            else:
                v = '{:.4f}'.format(model.parameters[i])
            lines.append('${} = {}$'.format(lname, v))

        # split lines into two columns
        n = len(model.param_names) - len(model.param_names) // 2
        figtext.append('\n'.join(lines[:n]))
        figtext.append('\n'.join(lines[n:]))

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

    # Adjust figure size for figtext
    if len(figtext) > 0:
        figsize = (figsize[0], figsize[1] + figtextsize)
    else:
        figtextsize = 0.

    # Create the figure
    fig = plt.figure(figsize=figsize)

    # Write figtext
    if len(figtext) > 0:
        for i in range(len(figtext)):
            if figtext[i] is None:
                continue
            xpos = 0.05 + 0.9 * (i / len(figtext))
            t = fig.text(xpos, 0.95, figtext[i],
                         va="top", ha="left", multialignment="left")

    # Loop over bands
    axnum = 0
    for wave, band in sorted(zip(waves, bands)):
        axnum += 1

        color = cmap((cm_wave_range[1] - wave) /
                     (cm_wave_range[1] - cm_wave_range[0]))

        ax = plt.subplot(nrow, ncol, axnum)
        if axnum % 2:
            plt.ylabel('flux ($ZP_{' + get_magsystem(zpsys).name.upper() +
                       '} = ' + str(zp) + '$)')

        xlabel_text = 'time'
        if len(models) > 0 and models[0].parameters[1] != 0.:
            xlabel_text += ' - {:.2f}'.format(models[0].parameters[1])

        # Plot data if there is any.
        if data is not None:
            idx = data_bands == band
            time = data['time'][idx]
            flux = data['flux'][idx]
            fluxerr = data['fluxerr'][idx]

            if len(models) == 0:
                plotted_time = time
            else:
                plotted_time = time - models[0].parameters[1]

            plt.errorbar(plotted_time, flux, fluxerr, ls='None',
                         color=color, marker='.', markersize=3.)

        # Plot model(s) if there are any.
        if len(models) > 0:
            mflux_mins = []
            mflux_maxes = []
            for i, model in enumerate(models):
                if not model.bandoverlap(band):
                    continue

                plotted_time = model.times - models[0].parameters[1]
                mflux = model.bandflux(band, zp=zp, zpsys=zpsys)

                if offsets is not None and band in offsets:
                    mflux = mflux + offsets[band]

                plt.plot(plotted_time, mflux, ls=_model_ls[i%len(_model_ls)],
                         marker='None', color=color, label=model.name)

                mflux_mins.append(mflux.min())
                mflux_maxes.append(mflux.max())

            # If we plotted any models, reset axes limits accordingly:
            if len(mflux_mins) > 0 and len(mflux_maxes) > 0:
                mflux_min = min(mflux_mins)
                mflux_max = max(mflux_maxes)
                ymin, ymax = ax.get_ylim()
                ymax = min(ymax, 2. * mflux_max)
                ymin = max(ymin, mflux_min - (ymax - mflux_max))
                ax.set_ylim(ymin, ymax)

            # Add a legend, if this is the first axes and there are two
            # or more models to distinguish between.
            if axnum == 1 and len(models) >= 2:
                leg = plt.legend(loc='upper right',
                                 fontsize='small', frameon=True)

        # Band name in corner: upper right if there is no legend, otherwise
        # upper left.
        if (axnum == 1 and len(models) > 1):
            plt.text(0.08, 0.92, band.name, color='k', ha='left', va='top',
                     transform=ax.transAxes)
        else:
            plt.text(0.92, 0.92, band.name, color='k', ha='right', va='top',
                     transform=ax.transAxes)

        # plot a horizontal line at flux=0.
        ax.axhline(y=0., ls='--', c='k')

        # steal part of the axes and plot pulls
        if (pulls and data is not None and len(models) == 1 and
            models[0].bandoverlap(band)):
            divider = make_axes_locatable(ax)
            axpulls = divider.append_axes("bottom", size=0.7, pad=0.1,
                                          sharex=ax)
            mflux = models[0].bandflux(band, time, zp=zp, zpsys=zpsys) 
            if offsets is not None and band in offsets:
                mflux = mflux + offsets[band]
            fluxpulls = (flux - mflux) / fluxerr
            plt.plot(time - models[0].parameters[1], fluxpulls, marker='.',
                     markersize=5., color=color, ls='None')
            plt.axhline(y=0., color=color)
            plt.setp(ax.get_xticklabels(), visible=False)
            if axnum % 2:
                plt.ylabel('pull')

        # label the most recent Axes x-axis
        plt.xlabel(xlabel_text)

    plt.tight_layout(rect=(0., 0., 1., 1. - figtextsize / figsize[1]))

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, **kwargs)
    plt.close()


# TODO remove this function.
@deprecated(since='0.4', message='use corner() from the triangle.py module: '
            'https://github.com/dfm/triangle.py')
def plot_param_samples(param_names, samples, weights=None, fname=None,
                       bins=25, panelsize=2.5, **kwargs):
    """Plot PDFs of parameter values.
    
    Parameters
    ----------
    param_names : list of str
        Parameter names.
    samples : `~numpy.ndarray` (nsamples, nparams)
        Parameter values.
    weights : `~numpy.ndarray` (nsamples)
        Weight of each sample.
    fname : str
        Output filename.
    bins : int
        Number of bins between -5*std and +5*std where std is the standard
        deviation of the samples for a given parameter.
    """
    from matplotlib import pyplot as plt
    from matplotlib.ticker import (NullFormatter, ScalarFormatter,
                                   NullLocator, AutoLocator)
    nullformatter = NullFormatter()
    formatter = ScalarFormatter()
    formatter.set_powerlimits((-2, 3))

    npar = len(param_names)

    # calculate average and std. dev. of each parameter
    avg = np.average(samples, weights=weights, axis=0)
    if weights is None:
        std = np.std(samples, axis=0)
    else:
        std = np.sqrt(np.sum(weights[:, np.newaxis] * samples**2, axis=0) -
                      avg**2)

    # Create figure
    figsize = (npar*panelsize, npar*panelsize)
    fig = plt.figure(figsize=figsize)

    for j in range(npar):
        ylims = (avg[j] - 5*std[j], avg[j] + 5*std[j])
        for i in range(j + 1):
            xlims = (avg[i] - 5*std[i], avg[i] + 5*std[i])

            ax = plt.subplot(npar, npar, j * npar + i + 1)

            # On diagonal, show a histogram.
            if i == j:
                plt.hist(samples[:, i], weights=weights, range=xlims,
                         bins=bins)

                # Write the average and standard deviation.
                text = '${0:s} = {1:s}$'.format(
                    param_names[i],
                    value_error_str(avg[i], std[i], latex=True)
                    )
                plt.text(0.9, 0.9, text, color='k', ha='right', va='top',
                         transform=ax.transAxes)

                # Make room for the text by pushing up the y limit.
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymax=1.2*ymax)

            # Otherwise, show a countour plot
            else:
                H, xedges, yedges = np.histogram2d(samples[:, i],
                                                   samples[:, j],
                                                   bins=bins,
                                                   weights=weights,
                                                   range=[xlims, ylims])
                X = 0.5 * (xedges[:-1] + xedges[1:])
                Y = 0.5 * (yedges[:-1] + yedges[1:])
                plt.contour(X, Y, H)
                plt.ylim(ylims)

            plt.xlim(xlims)

            # Tick locations
            xlocator = AutoLocator()
            xlocator.set_params(nbins=6)
            ax.xaxis.set_major_locator(xlocator)
            if i == j:
                ylocator = NullLocator()
            else:
                ylocator = AutoLocator()
                ylocator.set_params(nbins=6)
            ax.yaxis.set_major_locator(ylocator)

            # X axis labels & formatting
            if j < npar - 1:
                ax.xaxis.set_major_formatter(nullformatter)
            else:
                ax.xaxis.set_major_formatter(formatter)
                plt.xlabel(param_names[i])

            # Y axis labels & formatting
            if j == 0 or i > 0:
                ax.yaxis.set_major_formatter(nullformatter)
            else:
                ax.yaxis.set_major_formatter(formatter)
                plt.ylabel(param_names[j])

    plt.tight_layout()

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, **kwargs)
    plt.close()

def animate_model(model_or_models, fps=30, length=20.,
                  phase_range=(None, None), wave_range=(None, None),
                  match_peakphase=True, match_peakflux=True,
                  peakband='bessellb', fname=None, still=False):
    """Animate spectral timeseries of model(s) using matplotlib.animation.

    *Note:* Requires matplotlib v1.1 or higher.

    Parameters
    ----------
    model_or_models : `~sncosmo.SourceModel` or str or iterable thereof
        The model to animate or list of models to animate.
    fps : int, optional
        Frames per second. Default is 30.
    length : float, optional
        Movie length in seconds. Default is 15.
    phase_range : (float, float), optional
        Phase range to plot (in the timeframe of the first model if multiple
        models are given). `None` indicates to use the maximum extent of the
        model(s).
    wave_range : (float, float), optional
        Wavelength range to plot. `None` indicates to use the maximum extent
        of the model(s).
    match_peakflux : bool, optional
        For multiple models, scale fluxes so that the peak of the spectrum
        at the peak matches that of the first model. Default is
        True.
    match_peakphase : bool, optional
        For multiple models, shift additional models so that the model's
        reference phase matches that of the first model.
    peakband : `~sncosmo.Bandpass` or str, optional
        Bandpass used in match_peakflux and match_peakphase. Default is
        ``'bessellb'``.
    fname : str, optional
        If not `None`, save animation to file `fname`. Requires ffmpeg
        to be installed with the appropriate codecs: If `fname` has
        the extension '.mp4' the libx264 codec is used. If the
        extension is '.webm' the VP8 codec is used. Otherwise, the
        'mpeg4' codec is used. The first frame is also written to a
        png.
    still : bool, optional
        When writing to a file, also save the first frame as a png file.
        This is useful for displaying videos on a webpage.

    Examples
    --------
    Compare the salt2 and hsiao models:

    >>> animate_model(['salt2', 'hsiao'],  phase_range=(None, 30.),
    ...                                           # doctest: +SKIP
    ...               wave_range=(2000., 9200.))  # doctest: +SKIP

    Compare the salt2 model with ``x1=1`` to the same model with ``x1=0.``:

    >>> m1 = sncosmo.get_sourcemodel('salt2')  # doctest: +SKIP
    >>> m1.set(x1=1.)                          # doctest: +SKIP
    >>> m2 = sncosmo.get_sourcemodel('salt2')  # doctest: +SKIP
    >>> m2.set(x1=0.)                          # doctest: +SKIP
    >>> animate_model([m1, m2])                # doctest: +SKIP

    """

    from matplotlib import pyplot as plt
    from matplotlib import animation

    # Convert input to a list (if it isn't already).
    if (not isiterable(model_or_models) or
        isinstance(model_or_models, basestring)):
        models = [model_or_models]
    else:
        models = model_or_models

    # Check that all entries are SourceModel or strings.
    for m in models:
        if not (isinstance(m, basestring) or isinstance(m, SourceModel)):
            raise ValueError('str or SourceModel instance expected for '
                             'model(s)')
    models = [get_sourcemodel(m) for m in models]

    # Phase offsets needed to match peak phases.
    peakphases = [m.peakphase(peakband) for m in models]
    if match_peakphase:
        phase_offsets = [p - peakphases[0] for p in peakphases]
    else:
        phase_offsets =  [0.] * len(models)

    # Determine phase range to display.
    minphase, maxphase = phase_range
    if minphase is None:
        minphase = min([models[i].minphase - phase_offsets[i] for
                        i in range(len(models))])
    if maxphase is None:
        maxphase = max([models[i].maxphase - phase_offsets[i] for
                        i in range(len(models))])
    
    # Determine the wavelength range to display.
    minwave, maxwave = wave_range
    if minwave is None:
        minwave = min([m.minwave for m in models])
    if maxwave is None:
        maxwave = max([m.maxwave for m in models])

    # model time interval between frames
    phase_interval = (maxphase - minphase) / (length * fps)

    # maximum flux density of each model at the peak phase
    max_fluxes = [np.max(m.flux(phase))
                  for m, phase in zip(models, peakphases)]

    # scaling factors
    if match_peakflux:
        max_bandfluxes = [m.bandflux(peakband, phase)
                          for m, phase in zip(models, peakphases)]
        scaling_factors = [max_bandfluxes[0] / f for f in max_bandfluxes]
        global_max_flux = max_fluxes[0]
    else:
        scaling_factors = [1.] * len(models)
        global_max_flux = max(max_fluxes)

    ymin = -0.06 * global_max_flux
    ymax = 1.1 * global_max_flux

    # Pre-get wavelength array for each model.
    waves = [m.wavelengths for m in models]

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(minwave, maxwave), ylim=(ymin, ymax))
    plt.axhline(y=0., c='k')
    plt.xlabel('Wavelength ($\\AA$)') 
    plt.ylabel('Flux Density ($F_\lambda$)')
    phase_text = ax.text(0.05, 0.95, '', ha='left', va='top',
                        transform=ax.transAxes)
    empty_lists = 2 * len(models) * [[]]
    lines = ax.plot(*empty_lists, lw=1)
    for line, model in zip(lines, models):
        line.set_label(model.name)
    legend = plt.legend(loc='upper right')

    def init():
        for line in lines:
            line.set_data([], [])
        phase_text.set_text('')
        return tuple(lines) + (phase_text,)

    def animate(i):
        current_phase = minphase + phase_interval * i
        for j in range(len(models)):
            y = models[j].flux(current_phase + phase_offsets[j])
            lines[j].set_data(waves[j], y * scaling_factors[j])
        phase_text.set_text('phase = {:.1f}'.format(current_phase))
        return tuple(lines) + (phase_text,)

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=int(fps*length), interval=(1000./fps),
                                  blit=True)

    # Save the animation as an mp4 or webm file.
    # This requires that ffmpeg is installed.
    if fname is not None:
        if still:
            i = fname.rfind('.')
            stillfname = fname[:i] + '.png'
            plt.savefig(stillfname) 
        ext = fname[i+1:]
        codec = {'mp4': 'libx264', 'webm': 'libvpx'}.get(ext, 'mpeg4')
        ani.save(fname, fps=fps, codec=codec, extra_args=['-vcodec', codec],
                 writer='ffmpeg_file', bitrate=1800)
    else:
        plt.show()

    plt.close()
