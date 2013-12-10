# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Extinction functions."""

import os

import numpy as np
from scipy.ndimage import map_coordinates

import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits
from astropy.utils import isiterable
from astropy.config import ConfigurationItem

SFD_MAP_DIR = ConfigurationItem('sfd_map_dir', '.',
                                'Directory containing SFD (1998) dust maps, '
                                'with names: SFD_dust_4096_[ngp,sgp].fits')

__all__ = ['get_ebv_from_map']

def get_ebv_from_map(coordinates, mapdir=None, interpolate=True, order=1):
    """Get E(B-V) value(s) from Schlegel, Finkbeiner, and Davis 1998 extinction
    maps at the given coordinates.

    Parameters
    ----------
    coordinates : astropy `~astropy.coordinates.coordsystems.SphericalCoordinatesBase` or tuple/list.
        If tuple/list, treated as (RA, Dec) in degrees the ICRS (e.g., "J2000")
        system. RA and Dec can each be float or list or numpy array.
    mapdir : str, optional
        Directory in which to find dust map FITS images, which must be named
        ``SFD_dust_4096_[ngp,sgp].fits``. If `None` (default), the value of
        the SFD_MAP_DIR configuration item is used. By default, this is ``'.'``.
        The value of SFD_MAP_DIR can be set in the configuration file,
        typically located in ``$HOME/.astropy/config/sncosmo.cfg``.
    interpolate : bool
        Interpolate between the map values using
        `scipy.ndimage.map_coordinates`.
    order : int
        Interpolation order, if interpolate=True. Default is 1.

    Returns
    -------
    ebv : float or `~numpy.ndarray`
        Specific extinction E(B-V) at the given locations.

    Notes
    -----
    For large arrays of (RA, Dec) this function takes about 0.01 seconds
    per coordinate, where the runtime is (probably) dominated by the
    coordinate system conversion.
    """
    
    # Get mapdir
    if mapdir is None:
        mapdir = SFD_MAP_DIR()
    mapdir = os.path.expanduser(mapdir)
    mapdir = os.path.expandvars(mapdir)
    fname = os.path.join(mapdir, 'SFD_dust_4096_{0}.fits')

    # Parse input 
    if not isinstance(coordinates, coord.SphericalCoordinatesBase):
        ra, dec = coordinates
        coordinates = coord.ICRS(ra=ra, dec=dec, unit=(u.degree, u.degree))

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
    for n, idx, ext in [(1, b >= 0, 'ngp'), (-1, b < 0, 'sgp')]:

        if not np.any(idx): continue
        hdulist = fits.open(fname.format(ext))
        mapd = hdulist[0].data

        # Project from galactic longitude/latitude to lambert pixels.
        # (See SFD98).
        npix = mapd.shape[0]        
        x = (npix / 2 * np.cos(l[idx]) * np.sqrt(1. - n*np.sin(b[idx])) +
             npix / 2 - 0.5)
        y = (-npix / 2 * n * np.sin(l[idx]) * np.sqrt(1. - n*np.sin(b[idx])) +
             npix / 2 - 0.5)
        
        # Get map values at these pixel coordinates.
        if interpolate:
            ebv[idx] = map_coordinates(mapd, [y, x], order=order)
        else:
            x=np.round(x).astype(np.int)
            y=np.round(y).astype(np.int)
            ebv[idx] = mapd[y, x]
            
        hdulist.close()
    
    if return_scalar:
        return ebv[0]
    return ebv
