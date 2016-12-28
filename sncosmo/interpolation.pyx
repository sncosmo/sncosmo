#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""
mimic Grid2DFunction function in salt2 software snfit because it doesn't
use spline interpolation; it does bicubic convolution.
"""


import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport fabs


cdef int locate_below(double *array, int nvalues, double where):
    """Assumes a sorted array"""
    cdef:
        int low = 0
        int high = nvalues - 1
        int mid

    if nvalues == 0:
        return 0
    if (where > array[high]):
        return high
    if (where < array[low]):
        return -1
    while (high - low > 1):
        mid = (low+high)/2
        if (array[mid] > where):
            high = mid
        else:
            low = mid

    return low


cdef int locate_in_array(int lasti, double *xval, int nval, double xwhere):
    """Return i such that xval[i] <= xwhere < xval[i+1].

    Assumes that  xwhere is really between xval[0] and xval[nval-1]
    and the array is sorted.
    """
    cdef int i = lasti
  
    if (xwhere < xval[i]):
        i = locate_below(xval, i+1, xwhere)
    elif (xwhere >= xval[i+1]):
        if (i <= nval-3 and xwhere < xval[i+2]):
            i += 1;
        else:
            i = i + 1 + locate_below(xval + i + 1, nval - i - 1, xwhere)

    # note that if ( xval[i] <= xwhere < xval[i+2]), there is no binary search.
    if (i==nval-1 and xwhere == xval[nval-1]):
        i -= 1

    return i

cdef bint is_strictly_ordered(double[:] x):
    cdef int i
    for i in range(1, x.shape[0]):
        if x[i] <= x[i-1]:
            return 0
    return 1


DEF A = -0.5
DEF B = A + 2.0
DEF C = A + 3.0

cdef double kernval(double xval):
     cdef double x = fabs(xval)
     if x > 2.0:
         return 0.0
     if x < 1.0:
         return x * x * (B * x - C) + 1.0
     return A * (-4.0 + x * (8.0 + x * (-5.0 + x)))


cdef class BicubicInterpolator(object):
    """Equivalent of Grid2DFunction in snfit software.

    Bicubic convolution using kernel (and bilinear when next to edge). The
    kernel is defined by:

    x = fabs((distance of point to node)/(distance between nodes)) 

    W(x) = (a+2)*x**3-(a+3)*x**2+1 for x<=1
    W(x) = a( x**3-5*x**2+8*x-4) for 1<x<2
    W(x) = 0 for x>2
    """

    cdef double* xval
    cdef double* yval
    cdef double* fval_storage
    cdef double** fval
    cdef double xmin
    cdef double xmax
    cdef double ymin
    cdef double ymax
    cdef int nx
    cdef int ny

    def __cinit__(self, x, y, z):
        cdef:
            double[:] xc = np.asarray(x, dtype=np.float64)
            double[:] yc = np.asarray(y, dtype=np.float64)
            double[:,:] zc = np.asarray(z, dtype=np.float64)
            int i
            int j

        if not (is_strictly_ordered(xc) and is_strictly_ordered(yc)):
            raise ValueError("x and y values must be strictly increasing")

        self.nx = xc.shape[0]
        self.ny = yc.shape[0]

        # allocate xval
        self.xval = <double *>PyMem_Malloc(self.nx * sizeof(double))
        if not self.xval:
            raise MemoryError()
        for i in range(self.nx):
            self.xval[i] = xc[i]

        self.xmin = self.xval[0]
        self.xmax = self.xval[self.nx-1]

        # allocate yval
        self.yval = <double *>PyMem_Malloc(self.ny * sizeof(double))
        if not self.yval:
            raise MemoryError()
        for i in range(self.ny):
            self.yval[i] = yc[i]

        self.ymin = self.yval[0]
        self.ymax = self.yval[self.ny - 1]

        # copy values array
        self.fval_storage = <double *>PyMem_Malloc(self.nx * self.ny *
                                                   sizeof(double*))
        if not self.fval_storage:
            raise MemoryError()
        for i in range(self.nx):
            for j in range(self.ny):
                self.fval_storage[i * self.ny + j] = zc[i, j]

        # allocate fval: pointers to rows of main array
        self.fval = <double **>PyMem_Malloc(self.nx * sizeof(double*))
        if not self.fval:
            raise MemoryError()
        for i in range(self.nx):
            self.fval[i] = self.fval_storage + i * self.ny
        
    def __dealloc__(self):
        PyMem_Free(self.xval)
        PyMem_Free(self.yval)
        PyMem_Free(self.fval)
        PyMem_Free(self.fval_storage)

    def __call__(self, x, y):
        cdef:
            int i,j
            double[:, :] result_view
            double[:] xc = np.atleast_1d(np.asarray(x, dtype=np.float64))
            double[:] yc = np.atleast_1d(np.asarray(y, dtype=np.float64))
            double x_i, y_j
            int ix, iy
            int lastix = 0
            int lastiy = 0
            double ax, ay, ay2, dx, dy
            int nxc = xc.shape[0]
            int nyc = yc.shape[0]
            double *wyvec
            int *iyvec
            int *yflagvec
            int xflag
            double wx[4]

        # allocate result
        result = np.empty((nxc, nyc), dtype=np.float64)
        result_view = result

        # allocate and fill array of y indicies and weights
        # (could use static storage here for small vectors)
        wyvec = <double *>PyMem_Malloc(nyc * 4 * sizeof(double))
        iyvec = <int *>PyMem_Malloc(nyc * sizeof(int))

        # flags: -1 == "skip, return 0", 0 == "linear", 1 == "cubic"
        yflagvec = <int *>PyMem_Malloc(nyc * sizeof(int))

        # fill above three arrays with y value info
        for j in range(nyc):
            y_j = yc[j]

            # if y is out of range, we won't be using the value at all
            if (y_j < self.ymin or y_j > self.ymax):
                yflagvec[j] = -1
            else:
                iy = locate_in_array(lastiy, self.yval, self.ny, y_j)
                lastiy = iy
                iyvec[j] = iy

                # if we're too close to border, we will use linear
                # interpolation
                # so don't compute weights here
                if (self.ny < 3 or iy == 0 or iy > (self.ny - 3)):
                    yflagvec[j] = 0
                else:
                    # OK to use full cubic interpolation
                    yflagvec[j] = 1

                    # precompute weights
                    dy = ((self.yval[iy] - y_j) /
                          (self.yval[iy+1] - self.yval[iy]))
                    wyvec[4*j+0] = kernval(dy-1.0)
                    wyvec[4*j+1] = kernval(dy)
                    wyvec[4*j+2] = kernval(dy+1.0)
                    wyvec[4*j+3] = kernval(dy+2.0)

        # main loop
        for i in range(nxc):
            x_i = xc[i]

            # precompute some stuff for x
            if (x_i < self.xmin or x_i > self.xmax):
                xflag = -1
            else:
                ix = locate_in_array(lastix, self.xval, self.nx, x_i)
                lastix = ix
                if (self.nx < 3 or ix == 0 or ix > (self.nx - 3)):
                    xflag = 0
                else:
                    # OK to use full cubic interpolation
                    xflag = 1

                    # compute weights
                    dx = ((self.xval[ix] - x_i) /
                          (self.xval[ix+1] - self.xval[ix]))
                    wx[0] = kernval(dx-1.0)
                    wx[1] = kernval(dx)
                    wx[2] = kernval(dx+1.0)
                    wx[3] = kernval(dx+2.0)

            # innermost loop
            for j in range(nyc):
                yflag = yflagvec[j]

                # out-of-bounds: return 0.
                if xflag == -1 or yflag == -1:
                    result_view[i, j] = 0.0

                else:
                    iy = iyvec[j]
                    y_j = yc[j]
                    
                    # linear interpolation in *both* dimensions if *either* is
                    # too close to the border. This is how the original code
                    # works, so we mimic it here, even though its dumb.
                    if xflag == 0 or yflag == 0:

                        # If either dimension is 1, just return a close-ish
                        # value
                        if self.nx == 1 or self.ny == 1:
                            result_view[i, j] = self.fval[ix][iy]
                        else:
                            ax = ((x_i - self.xval[ix]) /
                                  (self.xval[ix+1] - self.xval[ix]))
                            ay = ((y_j - self.yval[iy]) /
                                  (self.yval[iy+1] - self.yval[iy]))
                            ay2 = 1.0 - ay
                            
                            result_view[i, j] = (
                                (1.0 - ax) * (ay2 * self.fval[ix  ][iy  ] +
                                              ay  * self.fval[ix  ][iy+1]) +
                                ax         * (ay2 * self.fval[ix+1][iy  ] +
                                              ay  * self.fval[ix+1][iy+1]))

                    # Full cubic convolution
                    else:
                        result_view[i, j] = (
                            wx[0] * (wyvec[4*j+0] * self.fval[ix-1][iy-1] +
                                     wyvec[4*j+1] * self.fval[ix-1][iy  ] +
                                     wyvec[4*j+2] * self.fval[ix-1][iy+1] +
                                     wyvec[4*j+3] * self.fval[ix-1][iy+2]) +
                            wx[1] * (wyvec[4*j+0] * self.fval[ix  ][iy-1] +
                                     wyvec[4*j+1] * self.fval[ix  ][iy  ] +
                                     wyvec[4*j+2] * self.fval[ix  ][iy+1] +
                                     wyvec[4*j+3] * self.fval[ix  ][iy+2]) +
                            wx[2] * (wyvec[4*j+0] * self.fval[ix+1][iy-1] +
                                     wyvec[4*j+1] * self.fval[ix+1][iy  ] +
                                     wyvec[4*j+2] * self.fval[ix+1][iy+1] +
                                     wyvec[4*j+3] * self.fval[ix+1][iy+2]) +
                            wx[3] * (wyvec[4*j+0] * self.fval[ix+2][iy-1] +
                                     wyvec[4*j+1] * self.fval[ix+2][iy  ] +
                                     wyvec[4*j+2] * self.fval[ix+2][iy+1] +
                                     wyvec[4*j+3] * self.fval[ix+2][iy+2]))

        PyMem_Free(iyvec)
        PyMem_Free(wyvec)
        PyMem_Free(yflagvec)

        return result
