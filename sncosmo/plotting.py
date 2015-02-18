# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to plot light curve data and models."""
from __future__ import division

import math

import numpy as np
from astropy.utils.misc import isiterable
from astropy.extern import six
from six.moves import range

from .models import Source, Model, get_source
from .spectral import get_bandpass, get_magsystem
from .photdata import standardize_data, normalize_data
from .utils import format_value

__all__ = ['plot_lc', 'animate_source']

_model_ls = ['-', '--', ':', '-.']


def plot_lc(data=None, model=None, bands=None, zp=25., zpsys='ab',
            pulls=True, xfigsize=None, yfigsize=None, figtext=None,
            model_label=None, errors=None, ncol=2, figtextsize=1.,
            show_model_params=True, tighten_ylim=False, color=None,
            cmap=None, cmap_lims=(3000., 10000.), fname=None, **kwargs):
    """Plot light curve data or model light curves.

    Parameters
    ----------
    data : astropy `~astropy.table.Table` or similar, optional
        Table of photometric data. Must include certain column names.
        See the "Photometric Data" section of the documentation for required
        columns.
    model : `~sncosmo.Model` or list thereof, optional
        If given, model light curve is plotted. If a string, the corresponding
        model is fetched from the registry. If a list or tuple of
        `~sncosmo.Model`, multiple models are plotted.
    model_label : str or list, optional
        If given, model(s) will be labeled in a legend in the upper left
        subplot. Must be same length as model.
    errors : dict, optional
        Uncertainty on model parameters. If given, along with exactly one
        model, uncertainty will be displayed with model parameters at the top
        of the figure.
    bands : list, optional
        List of Bandpasses, or names thereof, to plot.
    zp : float, optional
        Zeropoint to normalize the flux in the plot (for the purpose of
        plotting all observations on a common flux scale). Default is 25.
    zpsys : str, optional
        Zeropoint system to normalize the flux in the plot (for the purpose of
        plotting all observations on a common flux scale).
        Default is ``'ab'``.
    pulls : bool, optional
        If True (and if model and data are given), plot pulls. Pulls are the
        deviation of the data from the model divided by the data uncertainty.
        Default is ``True``.
    figtext : str, optional
        Text to add to top of figure. If a list of strings, each item is
        placed in a separate "column". Use newline separators for multiple
        lines.
    ncol : int, optional
        Number of columns of axes. Default is 2.
    xfigsize, yfigsize : float, optional
        figure size in inches in x or y. Specify one or the other, not both.
        Default is to set axes panel size to 3.0 x 2.25 inches.
    figtextsize : float, optional
        Space to reserve at top of figure for figtext (if not None).
        Default is 1 inch.
    show_model_params : bool, optional
        If there is exactly one model plotted, the parameters of the model
        are added to ``figtext`` by default (as two additional columns) so
        that they are printed at the top of the figure. Set this to False to
        disable this behavior.
    tighten_ylim : bool, optional
        If true, tighten the y limits so that the model is visible (if any
        models are plotted).
    color : str or mpl_color, optional
        Color of data and model lines in each band. Can be any type of color
        that matplotlib understands. If None (default) a colormap will be used
        to choose a color for each band according to its central wavelength.
    cmap : Colormap, optional
        A matplotlib colormap to use, if color is None. If both color
        and cmap are None, a default colormap will be used.
    cmap_lims : (float, float), optional
        The wavelength limits for the colormap, in Angstroms. Default is
        (3000., 10000.) meaning that a bandpass with a central wavelength of
        3000 Angstroms will be assigned a color at the low end of the colormap
        and a bandpass with a central wavelength of 10000 will be assigned a
        color at the high end of the colormap.
   fname : str, optional
        Filename to pass to savefig. If None (default), figure is returned.
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

    >>> import sncosmo
    >>> import matplotlib.pyplot as plt

    Load some example data:

    >>> data = sncosmo.load_example_data()

    Plot the data, displaying to the screen:

    >>> fig = plot_lc(data)
    >>> plt.show()  # doctest: +SKIP

    Plot a model along with the data:

    >>> model = sncosmo.Model('salt2')                # doctest: +SKIP
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
    from matplotlib.ticker import MaxNLocator, NullFormatter
    from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    if not all([isinstance(m, Model) for m in models]):
        raise TypeError('model(s) must be Model instance(s)')

    # Get the model labels
    if model_label is None:
        model_labels = [None] * len(models)
    elif isinstance(model_label, six.string_types):
        model_labels = [model_label]
    else:
        model_labels = model_label
    if len(model_labels) != len(models):
        raise ValueError('if given, length of model_label must match '
                         'that of model')

    # Color options.
    if color is None:
        if cmap is None:
            cmap = cm.get_cmap('jet_r')

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
    elif isinstance(figtext, six.string_types):
        figtext = [figtext]
    if len(models) == 1 and show_model_params:
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
        hpanel = 2.25
        wpanel = 3.
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
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

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
        tmin.append(model.mintime())
        tmax.append(model.maxtime())
    tmin = min(tmin)
    tmax = max(tmax)
    tgrid = np.linspace(tmin, tmax, int(tmax - tmin) + 1)

    # Loop over bands
    bands = list(bands)
    waves = [get_bandpass(b).wave_eff for b in bands]
    waves_and_bands = sorted(zip(waves, bands))
    for axnum in range(ncol * nrow):
        row = axnum // ncol
        col = axnum % ncol
        ax = axes[row, col]

        if axnum >= len(waves_and_bands):
            ax.set_visible(False)
            ax.set_frame_on(False)
            continue

        wave, band = waves_and_bands[axnum]

        bandname_coords = (0.92, 0.92)
        bandname_ha = 'right'
        if color is None:
            bandcolor = cmap((cmap_lims[1] - wave) /
                             (cmap_lims[1] - cmap_lims[0]))
        else:
            bandcolor = color

        # Plot data if there are any.
        if data is not None:
            mask = data['band'] == band
            time = data['time'][mask]
            flux = data['flux'][mask]
            fluxerr = data['fluxerr'][mask]
            ax.errorbar(time - toff, flux, fluxerr, ls='None',
                        color=bandcolor, marker='.', markersize=3.)

        # Plot model(s) if there are any.
        lines = []
        labels = []
        mflux_ranges = []
        for i, model in enumerate(models):
            if model.bandoverlap(band):
                mflux = model.bandflux(band, tgrid, zp=zp, zpsys=zpsys)
                mflux_ranges.append((mflux.min(), mflux.max()))
                l, = ax.plot(tgrid - toff, mflux,
                             ls=_model_ls[i % len(_model_ls)],
                             marker='None', color=bandcolor)
                lines.append(l)
            else:
                # Add a dummy line so the legend displays all models in the
                # first panel.
                lines.append(plt.Line2D([0, 1], [0, 1],
                                        ls=_model_ls[i % len(_model_ls)],
                                        marker='None', color=bandcolor))
            labels.append(model_labels[i])

        # Add a legend, if this is the first axes and there are two
        # or more models to distinguish between.
        if row == 0 and col == 0 and model_label is not None:
            leg = ax.legend(lines, labels, loc='upper right',
                            fontsize='small', frameon=True)
            bandname_coords = (0.08, 0.92)  # Move bandname to upper left
            bandname_ha = 'left'

        # Band name in corner
        ax.text(bandname_coords[0], bandname_coords[1], band,
                color='k', ha=bandname_ha, va='top', transform=ax.transAxes)

        ax.axhline(y=0., ls='--', c='k')  # horizontal line at flux = 0.
        ax.set_xlim((tmin-toff, tmax-toff))

        # If we plotted any models, narrow axes limits so that the model
        # is visible.
        if tighten_ylim and len(mflux_ranges) > 0:
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
            axpulls = divider.append_axes('bottom', size='30%', pad=0.15,
                                          sharex=ax)
            mflux = models[0].bandflux(band, time, zp=zp, zpsys=zpsys)
            fluxpulls = (flux - mflux) / fluxerr
            axpulls.axhspan(ymin=-1., ymax=1., color='0.95')
            axpulls.axhline(y=0., color=bandcolor)
            axpulls.plot(time - toff, fluxpulls, marker='.',
                         markersize=5., color=bandcolor, ls='None')

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

        # If this axes is one of the last `ncol`, set x label.
        # Otherwise don't show tick labels.
        if (len(bands) - axnum - 1) < ncol:
            if toff == 0.:
                bottomax.set_xlabel('time')
            else:
                bottomax.set_xlabel('time - {0:.2f}'.format(toff))
        else:
            for l in bottomax.get_xticklabels():
                l.set_visible(False)

    if fname is None:
        return fig
    plt.savefig(fname, **kwargs)
    plt.close()


def animate_source(source, label=None, fps=30, length=20.,
                   phase_range=(None, None), wave_range=(None, None),
                   match_peakphase=True, match_peakflux=True,
                   peakwave=4000., fname=None, still=False):
    """Animate spectral timeseries of model(s) using matplotlib.animation.

    *Note:* Requires matplotlib v1.1 or higher.

    Parameters
    ----------
    source : `~sncosmo.Source` or str or iterable thereof
        The Source to animate or list of sources to animate.
    label : str or list of str, optional
        If given, label(s) for Sources, to be displayed in a legend on
        the animation.
    fps : int, optional
        Frames per second. Default is 30.
    length : float, optional
        Movie length in seconds. Default is 15.
    phase_range : (float, float), optional
        Phase range to plot (in the timeframe of the first source if multiple
        sources are given). `None` indicates to use the maximum extent of the
        source(s).
    wave_range : (float, float), optional
        Wavelength range to plot. `None` indicates to use the maximum extent
        of the source(s).
    match_peakflux : bool, optional
        For multiple sources, scale fluxes so that the peak of the spectrum
        at the peak matches that of the first source. Default is
        True.
    match_peakphase : bool, optional
        For multiple sources, shift additional sources so that the source's
        reference phase matches that of the first source.
    peakwave : float, optional
        Wavelength used in match_peakflux and match_peakphase. Default is
        4000.
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

    Returns
    -------
    ani : `~matplotlib.animation.FuncAnimation`
        Animation object that can be shown or saved.

    Examples
    --------
    Compare the salt2 and hsiao sources:

    >>> import matplotlib.pyplot as plt
    >>> ani = animate_source(['salt2', 'hsiao'],  phase_range=(None, 30.),
    ...                      wave_range=(2000., 9200.))  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

    Compare the salt2 source with ``x1=1`` to the same source with ``x1=0.``:

    >>> m1 = sncosmo.get_source('salt2')  # doctest: +SKIP
    >>> m1.set(x1=1.)                     # doctest: +SKIP
    >>> m2 = sncosmo.get_source('salt2')  # doctest: +SKIP
    >>> m2.set(x1=0.)                     # doctest: +SKIP
    >>> ani = animate_source([m1, m2], label=['salt2, x1=1', 'salt2, x1=0'])
    ... # doctest: +SKIP
    >>> plt.show()                        # doctest: +SKIP
    """

    from matplotlib import pyplot as plt
    from matplotlib import animation

    # Convert input to a list (if it isn't already).
    if (not isiterable(source)) or isinstance(source, six.string_types):
        sources = [source]
    else:
        sources = source

    # Check that all entries are Source or strings.
    for m in sources:
        if not (isinstance(m, six.string_types) or isinstance(m, Source)):
            raise ValueError('str or Source instance expected for '
                             'source(s)')
    sources = [get_source(m) for m in sources]

    # Get the source labels
    if label is None:
        labels = [None] * len(sources)
    elif isinstance(label, six.string_types):
        labels = [label]
    else:
        labels = label
    if len(labels) != len(sources):
        raise ValueError('if given, length of label must match '
                         'that of source')

    # Get a wavelength array for each source.
    waves = [np.arange(m.minwave(), m.maxwave(), 10.) for m in sources]

    # Phase offsets needed to match peak phases.
    peakphases = [m.peakphase(peakwave) for m in sources]
    if match_peakphase:
        phase_offsets = [p - peakphases[0] for p in peakphases]
    else:
        phase_offsets = [0.] * len(sources)

    # Determine phase range to display.
    minphase, maxphase = phase_range
    if minphase is None:
        minphase = min([sources[i].minphase() - phase_offsets[i] for
                        i in range(len(sources))])
    if maxphase is None:
        maxphase = max([sources[i].maxphase() - phase_offsets[i] for
                        i in range(len(sources))])

    # Determine the wavelength range to display.
    minwave, maxwave = wave_range
    if minwave is None:
        minwave = min([m.minwave() for m in sources])
    if maxwave is None:
        maxwave = max([m.maxwave() for m in sources])

    # source time interval between frames
    phase_interval = (maxphase - minphase) / (length * fps)

    # maximum flux density of entire spectrum at the peak phase
    # for each source
    max_fluxes = [np.max(m.flux(phase, w))
                  for m, phase, w in zip(sources, peakphases, waves)]

    # scaling factors
    if match_peakflux:
        peakfluxes = [m.flux(phase, peakwave)  # Not the same as max_fluxes!
                      for m, phase in zip(sources, peakphases)]
        scaling_factors = [peakfluxes[0] / f for f in peakfluxes]
        global_max_flux = max_fluxes[0]
    else:
        scaling_factors = [1.] * len(sources)
        global_max_flux = max(max_fluxes)

    ymin = -0.06 * global_max_flux
    ymax = 1.1 * global_max_flux

    # Set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(minwave, maxwave), ylim=(ymin, ymax))
    plt.axhline(y=0., c='k')
    plt.xlabel('Wavelength ($\\AA$)')
    plt.ylabel('Flux Density ($F_\lambda$)')
    phase_text = ax.text(0.05, 0.95, '', ha='left', va='top',
                         transform=ax.transAxes)
    empty_lists = 2 * len(sources) * [[]]
    lines = ax.plot(*empty_lists, lw=1)
    if label is not None:
        for line, l in zip(lines, labels):
            line.set_label(l)
        legend = plt.legend(loc='upper right')

    def init():
        for line in lines:
            line.set_data([], [])
        phase_text.set_text('')
        return tuple(lines) + (phase_text,)

    def animate(i):
        current_phase = minphase + phase_interval * i
        for j in range(len(sources)):
            y = sources[j].flux(current_phase + phase_offsets[j], waves[j])
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
        plt.close()
    else:
        return ani
