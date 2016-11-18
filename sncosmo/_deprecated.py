# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Extinction functions."""

from __future__ import division

import os

import numpy as np
from scipy.ndimage import map_coordinates
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from astropy.io import fits
from astropy.utils import isiterable
from astropy.extern import six
from astropy.extern.six.moves import range

from .utils import warn_once
from .models import Source, get_source
from sncosmo import conf

__all__ = ['SFD98Map', 'get_ebv_from_map', 'animate_source']


class SFD98Map(object):
    """Map of E(B-V) from Schlegel, Finkbeiner and Davis (1998).

    This class is useful for repeated retrieval of E(B-V) values when
    there is no way to retrieve all the values at the same time: It keeps
    a reference to the FITS data from the maps so that each FITS image
    is read only once.  Note that there is still a large overhead due to
    coordinate conversion: When possible, pass arrays of coordinates to
    `SFD98Map.get_ebv` or `get_ebv_from_map`.

    Parameters
    ----------
    mapdir : str, optional
        Directory in which to find dust map FITS images, which must be
        named ``SFD_dust_4096_[ngp,sgp].fits``. If not specified,
        the value of the SFD_MAP_DIR configuration item is used. By
        default, this is ``'.'``.  The value of ``SFD_MAP_DIR`` can be set
        in the configuration file, typically located in
        ``$HOME/.astropy/config/sncosmo.cfg``.
    """

    def __init__(self, mapdir=None):

        warn_once("SFD98Map and get_ebv_from_map", "1.4", "2.0",
                  "Instead, use `SFDMap` and `ebv` from the sfdmap package: "
                  "http://github.com/kbarbary/sfdmap.")

        # Get mapdir
        if mapdir is None:
            mapdir = conf.sfd98_dir
        mapdir = os.path.expanduser(mapdir)
        mapdir = os.path.expandvars(mapdir)
        self.fname = os.path.join(mapdir, 'SFD_dust_4096_{0}.fits')

        # Don't load maps initially
        self.sgp = None
        self.ngp = None

    def get_ebv(self, coordinates, interpolate=True, order=1):
        """Get E(B-V) value(s) at given coordinate(s).

        Parameters
        ----------
        coordinates : astropy Coordinates object or tuple
            If tuple, treated as (RA, Dec) in degrees in the ICRS (e.g.,
            "J2000") system. RA and Dec can each be float or list or numpy
            array.
        interpolate : bool
            Interpolate between the map values using
            `scipy.ndimage.map_coordinates`. Default is ``True``.
        order : int
            Interpolation order, if interpolate=True. Default is ``1``.

        Returns
        -------
        ebv : float or `~numpy.ndarray`
            Specific extinction E(B-V) at the given locations.
        """

        # Parse input
        if not isinstance(coordinates, SkyCoord):
            ra, dec = coordinates
            coordinates = SkyCoord(ra=ra, dec=dec, frame='icrs', unit=u.degree)

        # Convert to galactic coordinates.
        coordinates = coordinates.galactic
        l = coordinates.l.radian
        b = coordinates.b.radian

        # Check if l, b are scalar. If so, convert to 1-d arrays.
        return_scalar = False
        if not isiterable(l):
            return_scalar = True
            l, b = np.array([l]), np.array([b])

        # Initialize return array
        ebv = np.empty_like(l)

        # Treat north (b>0) separately from south (b<0).
        for sign, mask, ext in [(1, b >= 0, 'ngp'), (-1, b < 0, 'sgp')]:
            if not np.any(mask):
                continue

            # Only load the FITS file for this hemisphere if it is needed
            # and has not been previously loaded. Once loaded, it will be
            # kept in memory for subsequent calls.
            if self.__dict__[ext] is None:
                hdulist = fits.open(self.fname.format(ext))
                header = hdulist[0].header
                self.__dict__[ext] = {'CRPIX1': header['CRPIX1'],
                                      'CRPIX2': header['CRPIX2'],
                                      'LAM_SCAL': header['LAM_SCAL'],
                                      'data': hdulist[0].data}
                hdulist.close()

            d = self.__dict__[ext]

            # Project from galactic longitude/latitude to lambert pixels.
            # (See SFD98).
            x = d['CRPIX1']-1. + (d['LAM_SCAL'] * np.cos(l[mask]) *
                                  np.sqrt(1. - sign*np.sin(b[mask])))
            y = d['CRPIX2']-1. - sign*(d['LAM_SCAL'] * np.sin(l[mask]) *
                                       np.sqrt(1. - sign*np.sin(b[mask])))

            # Get map values at these pixel coordinates.
            if interpolate:
                ebv[mask] = map_coordinates(d['data'], [y, x], order=order)
            else:
                x = np.round(x).astype(np.int)
                y = np.round(y).astype(np.int)
                ebv[mask] = d['data'][y, x]

        if return_scalar:
            return ebv[0]
        return ebv


def get_ebv_from_map(coordinates, mapdir=None, interpolate=True, order=1):
    """Get E(B-V) value(s) from Schlegel, Finkbeiner, and Davis (1998)
    extinction maps.

    Parameters
    ----------
    coordinates : astropy Coordinates object or tuple
        If tuple, treated as (RA, Dec) in degrees in the ICRS (e.g.,
        "J2000") system. RA and Dec can each be float or list or numpy
        array.
    mapdir : str, optional
        Directory in which to find dust map FITS images, which must be
        named ``SFD_dust_4096_[ngp,sgp].fits``. If `None` (default),
        the value of the ``sfd98_dir`` configuration item is used. By
        default, this is ``'.'``.  The value of ``sfd98_dir`` can be set
        in the configuration file, typically located in
        ``$HOME/.astropy/config/sncosmo.cfg``.
    interpolate : bool
        Interpolate between the map values using
        `scipy.ndimage.map_coordinates`. Default is ``True``.
    order : int
        Interpolation order used for interpolate=True. Default is 1.

    Returns
    -------
    ebv : float or `~numpy.ndarray`
        Specific extinction E(B-V) at the given locations.
    """

    m = SFD98Map(mapdir=mapdir)
    return m.get_ebv(coordinates, interpolate=interpolate, order=order)


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
    """

    from matplotlib import pyplot as plt
    from matplotlib import animation

    warn_once('animate_source', '1.4', '2.0')

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
