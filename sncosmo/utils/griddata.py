# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from copy import deepcopy

__all__ = ['GridData']

class GridData(object):
    """Interpolate over uniform 2-D grid.

    Similar to `scipy.interpolate.interp2d` but with methods for returning
    native sampling.
    
    Parameters
    ----------
    x0 : numpy.ndarray (1d)
    x1 : numpy.ndarray (1d)
    y : numpy.ndarray (2d)

    Examples
    --------
    Initialize the interpolator.

    >>> x0 = np.arange(-5., 5.01, 2.)
    >>> x1 = np.arange(-5., 5.01, 2.)
    >>> xx0, xx1 = np.meshgrid(x, y)
    >>> y = np.sin(xx0**2+xx1**2)
    >>> gd = GridData(x0, x1, y)

    Get the native sampling.

    >>> gd.x0
    array([-5., -3., -1.,  1.,  3.,  5.])
    >>> gd.x1
    array([-5., -3., -1.,  1.,  3.,  5.])

    Get interpolated values.

    >>> gd(0.)
    array([ 0.76255845, -0.54402111, 0.90929743, 0.90929743, -0.54402111,
           0.76255845])
    >>> gd(0., x1=[-2., 0., 2.])
    array([ 0.18263816, 0.90929743, 0.18263816])
    >>> gd.y(0., x1=-2)
    0.182638157968

    
    """
    
    def __init__(self, x0, x1, y):
        self._x0 = np.asarray(x0)
        self._x1 = np.asarray(x1)
        self._y = np.asarray(y)

        # Check shapes.
        if not (self._x0.ndim == 1 and self._x1.ndim == 1):
            raise ValueError("x0 and x1 must be 1-d")
        if not (self._y.ndim == 2):
            raise ValueError("y must be 2-d")
        if not self._y.shape == (len(self._x0), len(self._x1)):
            raise ValueError("y must have shape (len(x0), len(x1))")

    @property
    def x0(self):
        """Native x0 values."""
        return self._x0

    @property
    def x1(self):
        """Native x1 values."""
        return self._x1

    def __call__(self, x0, x1=None, extend=True):
        """Return y values at requested x0 and x1 values.

        Parameters
        ----------
        x0 : float
        x1 : numpy.ndarray, optional
            Default value is None, which is interpreted as native x1 values.
        extend : bool, optional
            The function raises ValueError if x0 is outside of native grid,
            unless extend is True, in which case it returns values at nearest
            native x0 value.

        Returns
        -------
        yvals : numpy.ndarray
            1-D array of interpolated y values at requested x0 value and
            x1 values.
        """

        # Bounds check first
        if (x0 < self._x0[0] or x0 > self._x0[-1]) and not extend:
            raise ValueError("Requested x0 {:.2f} out of range ({:.2f}, "
                             "{:.2f})".format(x0, self._x0[0], self._x0[-1]))

        # Use default x1 if none are specified
        if x1 is None: x1 = self._x1

        # Check if requested x0 is out of bounds or exactly in the list
        if (self._x0 == x0).any():
            idx = np.flatnonzero(self._x0 == x0)[0]
            return np.interp(x1, self._x1, self._y[idx, :])
        elif x0 < self._x0[0]:
            return np.interp(x1, self._x1, self._y[0, :])
        elif x0 > self._x0[-1]:
            return np.interp(x1, self._x1, self._y[-1, :])
            
        # If we got this far, we need to interpolate between x0 values
        i = np.searchsorted(self._x0, x0)
        y0 = np.interp(x1, self._x1, self._y[i - 1, :])
        y1 = np.interp(x1, self._x1, self._y[i, :])
        dx0 = ((x0 - self._x0[i - 1]) /
               (self._x0[i] - self._x0[i - 1]))
        dy = y1 - y0
        return y0 + dx0 * dy
