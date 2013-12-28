# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to plot light curve data and models."""
from __future__ import division

import math

import numpy as np
from astropy.utils.misc import isiterable
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

from .models import SourceModel, ObsModel, get_sourcemodel
from .spectral import get_bandpass, get_magsystem
from .photdata import standardize_data, normalize_data
from .utils import format_value

__all__ = ['plot_lc', 'animate_model']

_model_ls = ['-', '--', ':', '-.']
_cmap = cm.get_cmap('jet_r')
_cmap_wavelims = (3000., 10000.)

def plot_lc(data=None, model=None, bands=None, zp=25., zpsys='ab', pulls=True,
            xfigsize=None, yfigsize=None, figtext=None,
            errors=None, ncol=2, figtextsize=1., fname=None, **kwargs):
    """Plot light curve data or model light curves.

    Parameters
    ----------
    data : astropy `~astropy.table.Table` or similar
        Table of photometric data points.
    model : `~sncosmo.ObsModel` or list thereof
        If given, model light curve is plotted. If a string, the corresponding
        model is fetched from the registry. If a list or tuple of
        `~sncosmo.ObsModel`, multiple models are plotted.
    errors : dict, optional
        Uncertainty on model parameters. If given, along with exactly one
        model, uncertainty will be displayed with model parameters at the top
        of the figure.
    bands : list, optional
        List of Bandpasses, or names thereof, to plot.
    zp : float, optional
        Zeropoint to normalize the flux. Default is 25.
    zpsys : str or `~sncosmo.MagSystem`, optional
        Zeropoint system for `zp`. Default is 'ab'.
    pulls : bool, optional
        If True (and if model and data are given), plot pulls. Default is
        ``True``.
    figtext : str, optional
        Text to add to top of figure. If a list of strings, each item is
        placed in a separate "column". Use newline separators for multiple
        lines.
    ncol: int, optional
        Number of columns of axes. Default is 2.
    xfigsize, yfigsize : float, optional
        figure size in inches in x or y. Specify one or the other, not both.
        Default is to set axes panel size to 3.0 x 2.25 inches.
    figtextsize : float, optional
        Space to reserve at top of figure for figtext (if not None).
        Default is 1 inch.
    fname : str, optional
        Filename to pass to savefig. If `None` (default), figure is returned.
    kwargs : optional
        Any additional keyword args are passed to `~matplotlib.pyplot.savefig`.
        Popular options include ``dpi``, ``format``, ``transparent``. See
        matplotlib docs for full list.

    Returns
    -------
    fig : matplotlib `~matplotlib.figure.Figure`
        Only returned if `fname` is `None`. Display to screen with
        ``plt.show()`` or save with ``fig.savefig(filename)``. When creating
        many figures, be sure to close with ``plt.close(fig)``. 

    Examples
    --------
    Load some example data:

    >>> import sncosmo
    >>> data = sncosmo.load_example_data()

    Plot the data, displaying to the screen:

    >>> fig = sncosmo.plot_lc(data)
    >>> fig.show()  # doctest: + SKIP

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
        bands = set(bands)
    elif bands is None:
        bands = set(data['band'])
    else:
        bands = set(data['band']) & set(bands)

    # Build figtext (including model parameters, if there is exactly 1 model).
    if errors is None:
        errors = {}
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
            v = format_value(model.parameters[i], errors.get(name), latex=True)
            lines.append('${0} = {1}$'.format(lname, v))

        # Split lines into two columns.
        n = len(model.param_names) - len(model.param_names) // 2
        figtext.append('\n'.join(lines[:n]))
        figtext.append('\n'.join(lines[n:]))
    if len(figtext) == 0:
        figtextsize = 0.

    # Calculate layout of figure (columns, rows, figure size). We have to
    # calculate these explicitly because plt.tight_layout() doesn't space the
    # subplots as we'd like them when only some of them have xlabels/xticks.
    wspace = 0.6  # All in inches.
    hspace = 0.3
    lspace = 1.0
    bspace = 0.7
    trspace = 0.2
    nrow = (len(bands) - 1) // ncol + 1
    if xfigsize is None and yfigsize is None:
        hpanel = 3.
        wpanel = 2.25
    elif xfigsize is None:
        hpanel = (yfigsize - figtextsize - bspace - trspace -
                  hspace * (nrow - 1)) / nrow
        wpanel = hpanel * 3. / 2.25
    elif yfigsize is None:
        wpanel = (xfigsize - lspace - trspace - wspace * (ncol - 1)) / ncol
        hpanel = wpanel * 2.25 / 3.
    else:
        raise ValueError('cannot specify both xfigsize and yfigsize')
    figsize = (lspace + wpanel * ncol + wspace * (ncol - 1) + trspace,
               bspace + hpanel * nrow + hspace * (nrow - 1) + trspace +
               figtextsize)

    # Create the figure and axes.
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)

    fig.subplots_adjust(left=lspace / figsize[0],
                        bottom=bspace / figsize[1],
                        right=1. - trspace / figsize[0],
                        top=1. - (figtextsize + trspace) / figsize[1],
                        wspace=wspace / wpanel,
                        hspace=hspace / hpanel)

    # Write figtext at the top of the figure.
    for i, coltext in enumerate(figtext):
        if coltext is not None:
            xpos = (trspace / figsize[0] +
                    (1. - 2.*trspace/figsize[0]) * (i/len(figtext)))
            ypos = 1. - trspace / figsize[1]
            fig.text(xpos, ypos, coltext, va="top", ha="left",
                     multialignment="left")

    # If there is exactly one model, offset the time axis by the model's t0.
    if len(models) == 1 and data is not None:
        toff = models[0].parameters[1]
    else:
        toff = 0.

    # Global min and max of time axis.
    tmin, tmax = [], []
    if data is not None:
        tmin.append(np.min(data['time']) - 10.)
        tmax.append(np.max(data['time']) + 10.)
    for model in models:
        tmin.append(model.mintime)
        tmax.append(model.maxtime)
    tmin = min(tmin)
    tmax = max(tmax)
    tgrid = np.linspace(tmin, tmax, int(tmax - tmin) + 1)
            
    # Loop over bands
    bands = list(bands)
    waves = [get_bandpass(b).wave_eff for b in bands]
    waves_and_bands = sorted(zip(waves, bands))
    for axnum in xrange(ncol * nrow):
        row = axnum // ncol
        col = axnum % ncol
        ax = axes[row, col]

        if axnum >= len(waves_and_bands):
            ax.set_visible(False)
            ax.set_frame_on(False)
            continue

        wave, band = waves_and_bands[axnum]

        bandname_coords = (0.92, 0.92)
        color = _cmap((_cmap_wavelims[1] - wave) /
                      (_cmap_wavelims[1] - _cmap_wavelims[0]))

        # Plot data if there are any.
        if data is not None:
            mask = data['band'] == band
            time = data['time'][mask]
            flux = data['flux'][mask]
            fluxerr = data['fluxerr'][mask]
            ax.errorbar(time - toff, flux, fluxerr, ls='None',
                        color=color, marker='.', markersize=3.)

        # Plot model(s) if there are any.
        mflux_ranges = []
        for i, model in enumerate(models):
            if model.bandoverlap(band):
                mflux = model.bandflux(band, tgrid, zp=zp, zpsys=zpsys)
                mflux_ranges.append((mflux.min(), mflux.max()))
                ax.plot(tgrid - toff, mflux,
                        ls=_model_ls[i%len(_model_ls)],
                        marker='None', color=color, label=model.name)

        # Add a legend, if this is the first axes and there are two
        # or more models to distinguish between.
        if row == 0 and col == 0 and len(models) >= 2:
            leg = ax.legend(loc='upper right',
                            fontsize='small', frameon=True)
            bandname_coords = (0.08, 0.92)  # Move bandname to upper left

        # Band name in corner
        ax.text(bandname_coords[0], bandname_coords[1], band,
                color='k', ha='left', va='top', transform=ax.transAxes)

        ax.axhline(y=0., ls='--', c='k')  # horizontal line at flux = 0.
        ax.set_xlim((tmin-toff, tmax-toff))

        # If we plotted any models, narrow axes limits so that the model
        # is visible.
        if len(mflux_ranges) > 0:
            mfluxmin = min([r[0] for r in mflux_ranges])
            mfluxmax = max([r[1] for r in mflux_ranges])
            ymin, ymax = ax.get_ylim()
            ymax = min(ymax, 4. * mfluxmax)
            ymin = max(ymin, mfluxmin - (ymax - mfluxmax))
            ax.set_ylim(ymin, ymax)

        if col == 0:
            ax.set_ylabel('flux ($ZP_{{{0}}} = {1}$)'
                          .format(get_magsystem(zpsys).name.upper(), zp))

        show_pulls = (pulls and
                      data is not None and
                      len(models) == 1 and models[0].bandoverlap(band))

        # steal part of the axes and plot pulls
        if show_pulls:
            divider = make_axes_locatable(ax)
            axpulls = divider.append_axes('bottom', size='30%' , pad=0.15,
                                          sharex=ax)
            mflux = models[0].bandflux(band, time, zp=zp, zpsys=zpsys) 
            fluxpulls = (flux - mflux) / fluxerr
            axpulls.axhspan(ymin=-1., ymax=1., color='0.95')
            axpulls.axhline(y=0., color=color)
            axpulls.plot(time - toff, fluxpulls, marker='.',
                         markersize=5., color=color, ls='None')

            # Ensure y range is centered at 0.
            ymin, ymax = axpulls.get_ylim()
            absymax = max(abs(ymin), abs(ymax))
            axpulls.set_ylim((-absymax, absymax))

            # Set x limits to global values.
            axpulls.set_xlim((tmin-toff, tmax-toff))

            # Set small number of y ticks so tick labels don't overlap.
            axpulls.yaxis.set_major_locator(MaxNLocator(5))

            # Label the y axis and make sure ylabels align between axes.
            if col == 0:
                axpulls.set_ylabel('pull')
                axpulls.yaxis.set_label_coords(-0.75 * lspace / wpanel, 0.5)
                ax.yaxis.set_label_coords(-0.75 * lspace / wpanel, 0.5)

            # Set top axis ticks invisible
            for l in ax.get_xticklabels():
                l.set_visible(False)

            # Set ax to axpulls in order to adjust plots.
            bottomax = axpulls

        else:
            bottomax = ax

        # If this axes is one of the last `ncol`, set x label and rotate
        # tick labels. Otherwise don't show tick labels.
        if (len(bands) - axnum - 1) < ncol:
            if toff == 0.:
                bottomax.set_xlabel('time')
            else:
                bottomax.set_xlabel('time - {0:.2f}'.format(toff))
            #for l in bottomax.get_xticklabels():
            #    l.set_rotation(22.5)
        else:
            for l in bottomax.get_xticklabels():
                l.set_visible(False)

    if fname is None:
        return fig
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
        phase_text.set_text('phase = {0:.1f}'.format(current_phase))
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
