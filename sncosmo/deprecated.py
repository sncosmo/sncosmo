# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Extinction functions."""

import os
from warnings import warn

import numpy as np
from scipy.ndimage import map_coordinates
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from astropy.io import fits
from astropy.utils import isiterable

from sncosmo import conf

__all__ = ['SFD98Map', 'get_ebv_from_map']


warned = []


def warn_once(name, msg):
    global warned
    if name not in warned:
        warn(msg)
        warned.append(name)


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

    See Also
    --------
    get_ebv_from_map

    Examples
    --------

    >>> m = SFD98Map(mapdir='/path/to/SFD98/images')    # doctest: +SKIP

    Get E(B-V) value at RA, Dec = 0., 0.:

    >>> m.get_ebv((0., 0.))   # doctest: +SKIP
    0.031814847141504288

    Get E(B-V) at RA, Dec = (0., 0.) and (1., 0.):

    >>> m.get_ebv(([0., 1.], [0., 0.]))   # doctest: +SKIP
    array([ 0.03181485,  0.03275469])
    """

    def __init__(self, mapdir=None):

        warn_once("SFD98Map",
                  "`SFD98Map` and `get_ebv_from_map` are deprecated in "
                  "sncosmo v1.4 and will be removed in sncosmo v2.0. "
                  "Instead, use `SFDMap` and `ebv` from the sfdmap package; "
                  "see http://github.com/kbarbary/sfdmap.")

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

    Examples
    --------

    Get E(B-V) value at RA, Dec = (0., 0.):

    >>> get_ebv_from_map((0., 0.), mapdir='/path/to/dir')  # doctest: +SKIP
    0.031814847141504288

    Get E(B-V) at RA, Dec = (0., 0.) and (1., 0.):

    >>> get_ebv_from_map(([0., 1.], [0., 0.]), mapdir='/path/to/dir')
    ...                                                    # doctest: +SKIP
    array([ 0.03181485,  0.03275469])
    """

    m = SFD98Map(mapdir=mapdir)
    return m.get_ebv(coordinates, interpolate=interpolate, order=order)
