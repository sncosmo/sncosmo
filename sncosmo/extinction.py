# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Extinction functions."""

import os
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits
from astropy.config import ConfigurationItem as ConfigItem
from astropy.utils.misc import isiterable
from scipy.ndimage import map_coordinates

SFD_MAP_DIR = ConfigItem('sfd_map_dir', '.',
                         'Directory containing SFD (1998) dust maps, with '
                         'names: SFD_dust_4096_[ngp,sgp].fits')

__all__ = ['extinction_ccm', 'get_ebv_from_map']

# Optical/NIR coefficients from Cardelli (1989)
c1_ccm = [1., 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, -0.77530, 0.32999]
c2_ccm = [0., 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, 5.30260, -2.09002]
      
# Optical/NIR coefficents from O'Donnell (1994)
c1_odonnell = [1., 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505]
c2_odonnell = [0., 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347]

def extinction_ccm(wavelength, a_v=None, ebv=None, r_v=3.1,
                   optical_coeffs='odonnell'):
    r"""The Cardelli, Clayton, and Mathis (1989) extinction function.

    This function returns the total extinction A(\lambda) at the given
    wavelengths, given either the total V band extinction `a_v` or the
    selective extinction E(B-V) `ebv`, where `a_v = r_v * ebv`.

    Parameters
    ----------
    wavelength : float or list_like
        Wavelength(s) in Angstroms. Values must be between 909.1 and 33,333.3,
        the range of validity of the extinction curve paramterization.
    a_v or ebv: float
        Total V band extinction or selective extinction E(B-V) (must specify
        exactly one).
    r_v : float, optional
        Ratio of total to selective extinction: R_V = A_V / E(B-V).
        Default is 3.1.
    optical_coeffs : {'odonnell', 'ccm'}, optional
        If 'odonnell' (default), use the updated parameters for the optical
        given by O'Donnell (1994) [2]_. If 'ccm', use the original paramters
        given by Cardelli, Clayton and Mathis (1989) [1]_.

    Returns
    -------
    extinction_ratio : float or `~numpy.ndarray`
        Ratio of total to selective extinction: A(wavelengths) / E(B - V)
        at given wavelength(s). 

    Notes
    -----
    In [1]_ the mean :math:`R_V`-dependent extinction law, is parameterized
    as

    .. math::

       <A(\lambda)/A_V> = a(x) + b(x) / R_V

    where the coefficients a(x) and b(x) are functions of
    wavelength. At a wavelength of approximately 5494.5 Angstroms (a
    characteristic wavelength for the V band), a(x) = 1 and b(x) = 0,
    so that A(5494.5 Angstroms) = A_V. This function returns

    .. math::

       A(\lambda) = A_V * (a(x) + b(x) / R_V)

    where `A_V` can either be specified directly or via `E(B-V)`
    (by defintion, `A_V = R_V * E(B-V)`). The flux transmission fraction
    as a function of wavelength can then be obtained by

    .. math::

       T(\lambda) = (10^{-0.4 A(\lambda)})

    The extinction scales linearly with `a_v` or `ebv`, so one can compute
    t ahead of time for a given set of wavelengths with `a_v=1` and then
    scale by `a_v` later:
    `t_base = 10 ** (-0.4 * extinction_ccm(wavelengths, a_v=1.))`, then later:
    `t = np.power(t_base, a_v)`. Similarly for `ebv`.

    For an alternative to the CCM curve, see the extinction curve
    given in Fitzpatrick (1999) [6]_.

    **Notes from the IDL routine CCM_UNRED:**

    1. The CCM curve shows good agreement with the Savage & Mathis (1979)
       [5]_ ultraviolet curve shortward of 1400 A, but is probably
       preferable between 1200 and 1400 A.
    2. Many sightlines with peculiar ultraviolet interstellar extinction 
       can be represented with a CCM curve, if the proper value of 
       R(V) is supplied.
    3. Curve is extrapolated between 912 and 1000 A as suggested by
       Longo et al. (1989) [3]_.
    4. Valencic et al. (2004) [4]_ revise the ultraviolet CCM
       curve (3.3 -- 8.0 um-1).  But since their revised curve does
       not connect smoothly with longer and shorter wavelengths, it is
       not included here.


    References
    ----------
    .. [1] Cardelli, Clayton, and Mathis 1989, ApJ, 345, 245
    .. [2] O'Donnell 1994, ApJ, 422, 158
    .. [3] Longo et al. 1989, ApJ, 339,474
    .. [4] Valencic et al. 2004, ApJ, 616, 912
    .. [5] Savage & Mathis 1979, ARA&A, 17, 73
    .. [6] Fitzpatrick 1999, PASP, 111, 63
    """


    if (a_v is None) and (ebv is None):
        raise ValueError('Must specify either a_v or ebv')
    if (a_v is not None) and (ebv is not None):
        raise ValueError('Cannot specify both a_v and ebv')
    if a_v is None:
        a_v = r_v * ebv

    wavelength = np.asarray(wavelength)
    in_ndim = wavelength.ndim

    x = 1.e4 / wavelength.ravel()  # Inverse microns.
    if ((x < 0.3) | (x > 11.)).any():
        raise ValueError("extinction only defined in wavelength range"
                         " [909.091, 33333.3].")

    a = np.empty(x.shape, dtype=np.float)
    b = np.empty(x.shape, dtype=np.float)
  
    # Infrared
    idx = x < 1.1
    if idx.any():
        a[idx] = 0.574 * x[idx] ** 1.61
        b[idx] = -0.527 * x[idx] ** 1.61

    # Optical/NIR
    idx = (x >= 1.1) & (x < 3.3)
    if idx.any():
        xp = x[idx] - 1.82

        if optical_coeffs == 'odonnell':
            c1, c2 = c1_odonnell, c2_odonnell
        elif optical_coeffs == 'ccm':
            c1, c2 = c1_ccm, c2_ccm
        else:
            raise ValueError('Unrecognized optical_coeffs: {0!r}'
                             .format(optical_coeffs))

        # we need to flip the coefficients, because in polyval
        # c[0] corresponds to x^(N-1), but above, c[0] corresponds to x^0
        a[idx] = np.polyval(np.flipud(c1), xp)
        b[idx] = np.polyval(np.flipud(c2), xp)

    # Mid-UV
    idx = (x >= 3.3) & (x < 8.)
    if idx.any():
        xp = x[idx]
        a[idx] = 1.752 - 0.316 * xp - 0.104 / ((xp - 4.67)**2 + 0.341)
        b[idx] = -3.090 + 1.825 * xp + 1.206 / ((xp - 4.67)**2 + 0.263)

    idx = (x > 5.9) & (x < 8.)
    if idx.any():
        xp = x[idx] - 5.9
        a[idx] += -0.04473 * xp**2 - 0.009779 * xp**3
        b[idx] += 0.2130 * xp**2 + 0.1207 * xp**3

    # Far-UV
    idx = x >= 8.
    if idx.any():
        xp = x[idx] - 8.
        c1 = [ -1.073, -0.628,  0.137, -0.070]
        c2 = [ 13.670,  4.257, -0.420,  0.374]
        a[idx] = np.polyval(np.flipud(c1), xp)
        b[idx] = np.polyval(np.flipud(c2), xp)

    extinction = (a + b / r_v) * a_v
    if in_ndim == 0:
        return extinction[0]
    return extinction

def get_ebv_from_map(coordinates, map_dir=None, interpolate=True, order=1):
    """Get E(B-V) value(s) from Schlegel, Finkbeiner, and Davis 1998 extinction
    maps at the given coordinates.

    Parameters
    ----------
    coordinates : astropy Coordinates object or tuple
        If tuple/list, treated as (RA, Dec). RA and Dec can each be float or
        list or numpy array.
    map_dir : str, optional
        Directory in which to find dust map FITS images, which must be named
        ``SFD_dust_4096_[ngp,sgp].fits``. If `None` (default), the value of
        the SFD_MAP_DIR configuration item is used. By default, this is '.'.
        The value of SFD_MAP_DIR can be set in the configuration file,
        typically located in `$HOME/.astropy/config/sncosmo.cfg`.
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
    
    # Get map_dir
    if map_dir is None:
        map_dir = SFD_MAP_DIR()
    map_dir = os.path.expanduser(map_dir)
    map_dir = os.path.expandvars(map_dir)
    fname = os.path.join(map_dir, 'SFD_dust_4096_{}.fits')

    # Parse input 
    return_scalar = False
    if isinstance(coordinates, coord.SphericalCoordinatesBase):
        return_scalar = True
        coordinates = [coordinates]
    else:
        lon, lat = coordinates
        if not (isiterable(lon) or isiterable(lat)):
            return_scalar = True
            lon, lat = [lon], [lat]
        coordinates = [coord.ICRSCoordinates(ra=ra, dec=dec,
                                             unit=(u.degree, u.degree))
                       for ra, dec in zip(lon, lat)]

    # Convert to galactic coordinates (in radians).
    # Currently, coordinates do not support arrays; have to loop.
    l = np.empty(len(coordinates), dtype=np.float)
    b = np.empty(len(coordinates), dtype=np.float)
    for i, c in enumerate(coordinates):
        g = c.galactic

        # Hack to support both astropy v0.2.4 and v0.3.dev
        # TODO: remove this hack once v0.3 is out (and array-ify this
        # whole thing)
        try:
            l[i] = g.l.radian
            b[i] = g.b.radian
        except AttributeError:
            l[i] = g.l.radians
            b[i] = g.b.radians

    # Initialize return array
    ebv = np.zeros_like(l)

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
