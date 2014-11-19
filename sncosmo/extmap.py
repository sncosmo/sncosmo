# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Extinction functions."""

import os

import numpy as np
from scipy.ndimage import map_coordinates
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from astropy.io import fits
from astropy.utils import isiterable

from sncosmo import conf

__all__ = ['get_ebv_from_map']


def get_ebv_from_map(coordinates, mapdir=None, interpolate=True, order=1):

    """Get E(B-V) value(s) from Schlegel, Finkbeiner, and Davis (1998)
    extinction maps at the given coordinates.

    Parameters
    ----------
    coordinates : astropy Coordinates object or tuple/list.
        If tuple/list, treated as (RA, Dec) in degrees the ICRS (e.g.,
        "J2000") system. RA and Dec can each be float or list or numpy
        array.
    mapdir : str, optional
        Directory in which to find dust map FITS images, which must be
        named ``SFD_dust_4096_[ngp,sgp].fits``. If `None` (default),
        the value of the SFD_MAP_DIR configuration item is used. By
        default, this is ``'.'``.  The value of ``SFD_MAP_DIR`` can be set
        in the configuration file, typically located in
        ``$HOME/.astropy/config/sncosmo.cfg``.
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

    # Get mapdir
    if mapdir is None:
        mapdir = conf.sfd98_dir
    mapdir = os.path.expanduser(mapdir)
    mapdir = os.path.expandvars(mapdir)
    fname = os.path.join(mapdir, 'SFD_dust_4096_{0}.fits')

    # Parse input
    if not isinstance(coordinates, SkyCoord):
        ra, dec = coordinates
        coordinates = SkyCoord(ra=ra, dec=dec, frame='icrs', unit=u.degree)

    # Convert to galactic coordinates.
    coordinates = coordinates.galactic
    l = coordinates.l.radian
    b = coordinates.b.radian

    # Check if l, b are scalar
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

        hdulist = fits.open(fname.format(ext))
        header = hdulist[0].header
        data = hdulist[0].data

        # Project from galactic longitude/latitude to lambert pixels.
        # (See SFD98).
        x = header['CRPIX1']-1. + (header['LAM_SCAL'] * np.cos(l[mask]) *
                                   np.sqrt(1. - sign*np.sin(b[mask])))
        y = header['CRPIX2']-1. - sign*(header['LAM_SCAL'] * np.sin(l[mask]) *
                                        np.sqrt(1. - sign*np.sin(b[mask])))


        # Get map values at these pixel coordinates.
        if interpolate:
            ebv[mask] = map_coordinates(data, [y, x], order=order)
        else:
            x = np.round(x).astype(np.int)
            y = np.round(y).astype(np.int)
            ebv[mask] = data[y, x]

        hdulist.close()

    if return_scalar:
        return ebv[0]
    return ebv
