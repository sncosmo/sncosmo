# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to plot light curve data and models."""
from __future__ import division

import math

import numpy as np
from astropy.extern import six
from astropy.extern.six.moves import range

from .models import Model
from .bandpasses import get_bandpass
from .magsystems import get_magsystem
from .photdata import photometric_data
from .utils import format_value

__all__ = ['plot_lc']

_model_ls = ['-', '--', ':', '-.']


def _add_errorbar(ax, x, y, yerr, filled, markersize=None, color=None):
    """Add an errorbar to Axes `ax`, allowing an array of markers."""
    ax.errorbar(x[filled], y[filled], yerr[filled], ls='None',
                marker='o', markersize=markersize, color=color)
    notfilled = ~filled
    ax.errorbar(x[notfilled], y[notfilled], yerr[notfilled], ls='None',
                mfc='None', marker='o', markersize=markersize, color=color)


def _add_plot(ax, x, y, filled, markersize=None, color=None):
    ax.plot(x[filled], y[filled], marker='o',
            markersize=markersize, color=color, ls='None')
    notfilled = ~filled
    ax.plot(x[notfilled], y[notfilled], marker='o', mfc='None',
            markersize=markersize, color=color, ls='None')


def plot_lc(data=None, model=None, bands=None, zp=25., zpsys='ab',
            pulls=True, xfigsize=None, yfigsize=None, figtext=None,
            model_label=None, errors=None, ncol=2, figtextsize=1.,
            show_model_params=True, tighten_ylim=False, color=None,
            cmap=None, cmap_lims=(3000., 10000.), fill_data_marker=None,
            fname=None, fill_percentiles=None, **kwargs):
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
        (3000., 10000.), meaning that a bandpass with a central wavelength of
        3000 Angstroms will be assigned a color at the low end of the colormap
        and a bandpass with a central wavelength of 10000 will be assigned a
        color at the high end of the colormap.
    fill_data_marker : array_like, optional
        Array of booleans indicating whether to plot a filled or unfilled
        marker for each data point. Default is all filled markers.
    fname : str, optional
        Filename to pass to savefig. If None (default), figure is returned.
    fill_percentiles : (float, float, float), optional
        When multiple models are given, the percentiles for a light
        curve confidence interval. The upper and lower perceniles
        define a fill between region, and the middle percentile
        defines a line that will be plotted over the fill between
        region.
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
    >>> plt.show()

    Plot a model along with the data:

    >>> model = sncosmo.Model('salt2')
    >>> model.set(z=0.5, c=0.2, t0=55100., x0=1.547e-5)
    >>> sncosmo.plot_lc(data, model=model)

    .. image:: /pyplots/plotlc_example.png

    Plot just the model, for selected bands:

    >>> sncosmo.plot_lc(model=model,
    ...                 bands=['sdssg', 'sdssr'])

    Plot figures on a multipage pdf:

    >>> from matplotlib.backends.backend_pdf import PdfPages
    >>> pp = PdfPages('output.pdf')

    >>> # Do the following as many times as you like:
    >>> sncosmo.plot_lc(data, fname=pp, format='pdf')

    >>> # Don't forget to close at the end:
    >>> pp.close()

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
        data = photometric_data(data)
        data = data.normalized(zp=zp, zpsys=zpsys)
        if not np.all(np.ediff1d(data.time) >= 0.0):
            sortidx = np.argsort(data.time)
            data = data[sortidx]
        else:
            sortidx = None

    # Bands to plot
    if data is None:
        bands = set(bands)
    elif bands is None:
        bands = set(data.band)
    else:
        bands = set(data.band) & set(bands)

    # ensure bands is a list of Bandpass objects
    bands = [get_bandpass(b) for b in bands]

    # filled: used only if data is not None. Guarantee array of booleans
    if data is not None:
        if fill_data_marker is None:
            fill_data_marker = np.ones(data.time.shape, dtype=np.bool)
        else:
            fill_data_marker = np.asarray(fill_data_marker)
            if fill_data_marker.shape != data.time.shape:
                raise ValueError("fill_data_marker shape does not match data")
        if sortidx is not None:  # sort like we sorted the data
            fill_data_marker = fill_data_marker[sortidx]

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
        tmin.append(np.min(data.time) - 10.)
        tmax.append(np.max(data.time) + 10.)
    for model in models:
        tmin.append(model.mintime())
        tmax.append(model.maxtime())
    tmin = min(tmin)
    tmax = max(tmax)
    tgrid = np.linspace(tmin, tmax, int(tmax - tmin) + 1)

    # Loop over bands
    waves = [b.wave_eff for b in bands]
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
            mask = data.band == band
            time = data.time[mask]
            flux = data.flux[mask]
            fluxerr = data.fluxerr[mask]
            bandfilled = fill_data_marker[mask]
            _add_errorbar(ax, time - toff, flux, fluxerr, bandfilled,
                          color=bandcolor, markersize=3.)

        # Plot model(s) if there are any.
        lines = []
        labels = []
        mflux_ranges = []
        mfluxes = []
        plotci = len(models) > 1 and fill_percentiles is not None

        for i, model in enumerate(models):
            if model.bandoverlap(band):
                mflux = model.bandflux(band, tgrid, zp=zp, zpsys=zpsys)
                if not plotci:
                    mflux_ranges.append((mflux.min(), mflux.max()))
                    l, = ax.plot(tgrid - toff, mflux,
                                 ls=_model_ls[i % len(_model_ls)],
                                 marker='None', color=bandcolor)
                    lines.append(l)
                else:
                    mfluxes.append(mflux)
            else:
                # Add a dummy line so the legend displays all models in the
                # first panel.
                lines.append(plt.Line2D([0, 1], [0, 1],
                                        ls=_model_ls[i % len(_model_ls)],
                                        marker='None', color=bandcolor))
            labels.append(model_labels[i])

        if plotci:
            lo, med, up = np.percentile(mfluxes, fill_percentiles, axis=0)
            l, = ax.plot(tgrid - toff, med, marker='None',
                         color=bandcolor)
            lines.append(l)
            ax.fill_between(tgrid - toff, lo, up, color=bandcolor,
                            alpha=0.4)

        # Add a legend, if this is the first axes and there are two
        # or more models to distinguish between.
        if row == 0 and col == 0 and model_label is not None:
            leg = ax.legend(lines, labels, loc='upper right',
                            fontsize='small', frameon=True)
            bandname_coords = (0.08, 0.92)  # Move bandname to upper left
            bandname_ha = 'left'

        # Band name in corner
        text = band.name if band.name is not None else str(band)
        ax.text(bandname_coords[0], bandname_coords[1], text,
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
            _add_plot(axpulls, time - toff, fluxpulls, bandfilled,
                      markersize=4., color=bandcolor)

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
