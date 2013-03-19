# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

__all__ = ['GridData1d', 'GridData2d']

class GridData2d(object):
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
    >>> gd = GridData2d(x0, x1, y)

    Get the native sampling.

    >>> gd.x0
    array([-5., -3., -1.,  1.,  3.,  5.])
    >>> gd.x1
    array([-5., -3., -1.,  1.,  3.,  5.])

    Get interpolated values.

    >>> gd(0.)
    array([ 0.76255845, -0.54402111, 0.90929743, 0.90929743, -0.54402111,
           0.76255845])
    >>> gd(0., [-2., 0., 2.])
    array([ 0.18263816, 0.90929743, 0.18263816])
    >>> gd(0., -2.)
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

    def __call__(self, x0=None, x1=None, extend=False):
        """Return y values at requested x0 and x1 values.

        Parameters
        ----------
        x0 : float or `~numpy.ndarray`, optional
            Default is `None`, which is interpreted as the native values.
        x1 : float or `~numpy.ndarray`, optional
            Default is `None`, which is interpreted as the native values.
        extend : bool, optional
            The function raises ValueError if x0 is outside of native grid,
            unless extend is True, in which case it returns values at nearest
            native x0 value. Default is False.

        Returns
        -------
        yvals : numpy.ndarray
            1-D array of interpolated y values at requested x0 value and
            x1 values.
        """

        if x0 is None and x1 is None:
            return self._y

        if x0 is None: x0 = self._x0
        else: x0 = np.asarray(x0)
        if x1 is None: x1 = self._x1
        else: x1 = np.asarray(x1)            
        if x0.ndim > 1 or x1.ndim > 1:
                raise ValueError("x0 and x1 can be at most 1-d")

        in_dims = (x0.ndim, x1.ndim)
        x0 = x0.ravel()
        x1 = x1.ravel()

        if not extend:
            if (x0[0] < self._x0[0] or x0[-1] > self._x0[-1]):
                raise ValueError("x0 out of range ({:f}, {:f})"
                                 .format(self._x0[0], self._x0[-1]))
            if (x1[0] < self._x1[0] or x1[-1] > self._x1[-1]):
                raise ValueError("x1 out of range ({:f}, {:f})"
                                 .format(self._x1[0], self._x1[-1]))

        y = np.empty((x0.shape[0], x1.shape[0]))
        for i, j in enumerate(np.searchsorted(self._x0, x0)):
            if j == 0:
                y[i, :] = np.interp(x1, self._x1, self._y[0, :])
            elif j == self._x0.shape[0]:
                y[i, :] = np.interp(x1, self._x1, self._y[-1, :])
            else:
                y0 = np.interp(x1, self._x1, self._y[j - 1, :])
                y1 = np.interp(x1, self._x1, self._y[j, :])
                dx0 = (x0[i] - self._x0[j-1]) / (self._x0[j] - self._x0[j-1])
                y[i, :] = y0 + dx0 * (y1 - y0)

        if in_dims[0] == 0:
            if in_dims[1] == 0:
                return y[0, 0]
            return y[0]
        return y


class GridData1d(object):
    """Interpolate over uniform 1-D grid.

    Similar to `numpy.interp`, but it is a class.

    Parameters
    ----------
    x : numpy.ndarray (1d)
    y : numpy.ndarray (1d)
    """

    def __init__(self, x, y):
        self._x = np.asarray(x)
        self._y = np.asarray(y)

        if not (self._x.ndim == 1):
            raise ValueError("x must be 1-d")
        if not (self._y.shape == self._x.shape):
            raise ValueError("x and y must have same shape")

    @property
    def x(self):
        """Native x values."""
        return self._x

    def __call__(self, x=None, extend=False):
        """Return y values at requested x value.

        Parameters
        ----------
        x : float or `~numpy.ndarray`, optional
            Default is `None`, which is interpreted as the native values.
        extend : bool, optional
            The function raises ValueError if x is outside of native grid,
            unless extend is True, in which case it returns values at nearest
            native x value. Default is False.

        Returns
        -------
        y : numpy.ndarray
            1-D array of interpolated y values at requested x value.
        """

        if x is None:
            return self._y

        x = np.asarray(x)
        in_dim = x.ndim
        x = x.ravel()

        if not extend and (x[0] < self._x[0] or x[-1] > self._x[-1]):
            raise ValueError("x out of range ({:f}, {:f})"
                             .format(self._x[0], self._x[-1]))

        y = np.interp(x, self._x, self._y)

        if in_dim == 0: return y[0]
        return y
