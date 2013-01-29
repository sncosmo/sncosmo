# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

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

    >>> gd.x0()
    array([-5., -3., -1.,  1.,  3.,  5.])
    >>> gd.x1()
    array([-5., -3., -1.,  1.,  3.,  5.])

    Get interpolated values.

    >>> gd.y(0.)
    array([-0.26237485,  0.52908269,  0.76255845,  0.76255845,  0.52908269,
           -0.26237485])
    >>> gd.y(0., x1=[-2., 0., 2.])
    array([ 0.64582057,  0.76255845,  0.64582057])
    >>> gd.y(0., x1=-2)
    0.64582056829981327
    
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

        self._yfuncs = []
        for i in range(len(self._x0)):
            self._yfuncs.append(lambda x: np.interp(x, self._x1, y[i,:]))

    def x0(self, copy=False):
        """Native x0 values."""
        if copy: return self._x0.copy()
        else: return self._x0

    def x1(self, copy=False):
        """Native x1 values."""
        if copy: return self._x1.copy()
        else: return self._x1

    def y(self, x0, x1=None, extend=True):
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
            return self._yfuncs[idx](x1)
        elif x0 < self._x0[0]:
            return self._yfuncs[0](x1)
        elif x0 > self._x0[-1]:
            return self._yfuncs[-1](x1)
            
        # If we got this far, we need to interpolate between x0 values
        i = np.searchsorted(self._x0, x0)
        y0 = self._yfuncs[i - 1](x1)
        y1 = self._yfuncs[i](x1)
        dx0 = ((x0 - self._x0[i - 1]) /
               (self._x0[i] - self._x0[i - 1]))
        dy = y1 - y0
        return y0 + dx0 * dy
