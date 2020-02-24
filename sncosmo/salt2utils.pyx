# cython: boundscheck=False, wraparound=False, initializedcheck=False
# cython: cdivision=True, auto_pickle=False, language_level=3
# (need auto_pickle=False because we've implemented explicit pickle support via getnewargs() below)
"""
mimic Grid2DFunction function in salt2 software snfit because it doesn't
use spline interpolation; it does bicubic convolution.
"""


import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport fabs
from libc.string cimport memcpy


cdef int find_index_binary(double *values, int n, double x):
    """Find index i in array such that values[i] <= x < values[i+1].
    using binary search.

    Guaranteed to return values between 0 and n-2 inclusive.
    """
    
    cdef int lo, hi, mid

    lo = 0
    hi = n
    mid = n/2

    if (x < values[0]):
        return 0
    if (x >= values[n-1]):
        return n-2

    while (hi - lo > 1):
        if (x >= values[mid]):
            lo = mid
        else:
            hi = mid
        mid = lo + (hi - lo) / 2

    return mid



cdef int find_index_unsafe(double *values, int n, double x, int start):
    """Return i such that values[i] <= x < values[i+1] via linear search,
    starting from guess `start`.

    If x == values[n-1], n-2 is returned instead of n-1.

    This *assumes* that values[0] <= x <= values[n-1], and that
    0 <= start <= n-2. If that's not true, this will segfault.
    """
    cdef int i

    # search up
    if (x >= values[start]):
        i = start + 1
        while (i < n and x >= values[i]):
            i += 1
        if i == n:
            i -= 1
        return i-1

    # search down
    else:
        i = start - 1
        while (i > -1 and x < values[i]):
            i -= 1
        return i  # -1 should never be returned b/c we assume x >= values[0]


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
                                                   sizeof(double))
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
            int ix = 0
            int iy = 0
            double ax, ay, ay2, dx, dy
            int nxc = xc.shape[0]
            int nyc = yc.shape[0]
            double *wyvec
            int *iyvec
            int *yflagvec
            int xflag
            double *wx = [0., 0., 0., 0.]

        # allocate result
        result = np.empty((nxc, nyc), dtype=np.float64)
        result_view = result

        # allocate arrays of y indicies and weights
        # (could use static storage here for small vectors)
        wyvec = <double *>PyMem_Malloc(nyc * 4 * sizeof(double))
        iyvec = <int *>PyMem_Malloc(nyc * sizeof(int))

        # flags: -1 == "skip, return 0", 0 == "linear", 1 == "cubic"
        yflagvec = <int *>PyMem_Malloc(nyc * sizeof(int))

        # find initial indicies by binary search, because they could be
        # anywhere.
        if nxc > 0:
            ix = find_index_binary(self.xval, self.nx, xc[0])
        if nyc > 0:
            iy = find_index_binary(self.yval, self.ny, yc[0])

        # fill above three arrays with y value info
        for j in range(nyc):
            y_j = yc[j]

            # if y is out of range, we won't be using the value at all
            if (y_j < self.ymin or y_j > self.ymax):
                yflagvec[j] = -1
            else:
                iy = find_index_unsafe(self.yval, self.ny, y_j, iy)
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
                ix = find_index_unsafe(self.xval, self.nx, x_i, ix)
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

    def __getnewargs__(self):
        """Return arguments to pass to constructor (to support pickling)"""

        cdef:
            np.ndarray[np.double_t, ndim=1] x
            np.ndarray[np.double_t, ndim=1] y
            np.ndarray[np.double_t, ndim=2] z

        x = np.empty(self.nx, dtype=np.float64)
        y = np.empty(self.ny, dtype=np.float64)
        z = np.empty((self.nx, self.ny), dtype=np.float64)

        memcpy(&x[0], self.xval, self.nx * sizeof(double))
        memcpy(&y[0], self.yval, self.ny * sizeof(double))
        memcpy(&z[0,0], self.fval_storage, self.nx * self.ny * sizeof(double))

        return x, y, z


cdef double polyval(double *coeffs, int n, double x):
    "coeffs[0]*x + coeffs[1]*x^2 + ... + coeffs[n-1]*x^ncoeffs"""

    cdef double out = 0.0
    while n > 0:
        n -= 1
        out = x * (coeffs[n] + out)

    return out


# constants used in SALT2ColorLaw
DEF SALT2CL_B = 4302.57  # B-band-ish wavelength
DEF SALT2CL_V = 5428.55  # V-band-ish wavelength
DEF SALT2CL_V_MINUS_B = SALT2CL_V - SALT2CL_B


cdef class SALT2ColorLaw(object):
    """Callable returning extinction in magnitudes for c=1.

    This is the version 1 extinction law used in SALT2 2.0 (SALT2-2-0)
    and later.

    Parameters
    ----------
    wave_range : (float, float)
    coeffs : list_like


    Notes
    -----
    From snfit code comments:

    if(l_B<=l<=l_R):
        ext = exp(color * constant *
                  (alpha*l + params(0)*l^2 + params(1)*l^3 + ... ))
            = exp(color * constant * P(l))

        where alpha = 1 - params(0) - params(1) - ...

    if (l > l_R):
        ext = exp(color * constant * (P(l_R) + P'(l_R) * (l-l_R)))
    if (l < l_B):
        ext = exp(color * constant * (P(l_B) + P'(l_B) * (l-l_B)))
    """

    cdef:
        int ncoeffs
        double coeffs[7]  # can store up to 6 coeffs (should be only 4)
        double l_lo
        double l_hi
        double p_lo
        double p_hi
        double pprime_lo
        double pprime_hi

    def __cinit__(self, wave_range, coeffs):
        cdef:
            int i
            double wave_lo
            double wave_hi
            double[:] ccoeffs = np.asarray(coeffs, dtype=np.float64)
            double dcoeffs[6]

        if ccoeffs.shape[0] > 6:
            raise ValueError("number of coefficients must be equal to or "
                             "less than 6.")

        # set wave_range
        wave_lo, wave_hi = wave_range
        self.l_lo = (wave_lo - SALT2CL_B) / SALT2CL_V_MINUS_B
        self.l_hi = (wave_hi - SALT2CL_B) / SALT2CL_V_MINUS_B

        for i in range(ccoeffs.shape[0]):
            self.coeffs[i+1] = ccoeffs[i]

        # first coefficient is 'alpha' = 1.0 - sum(other coeffs)
        self.ncoeffs = ccoeffs.shape[0] + 1
        self.coeffs[0] = 1.0
        for i in range(1, self.ncoeffs):
            self.coeffs[0] -= self.coeffs[i]

        # precompute value of
        # P(l) = c[0]*l + c[1]*l^2 + c[2]*l^3 + ...  at l_lo and l_hi
        self.p_lo = polyval(self.coeffs, self.ncoeffs, self.l_lo)
        self.p_hi = polyval(self.coeffs, self.ncoeffs, self.l_hi)

        # precompute derivative of P(l) at l_lo and l_hi
        # P'(l) = c[0] + 2*c[1]*l + 3*c[2]*l^2 + ...)
        for i in range(self.ncoeffs-1):
            dcoeffs[i] = (i+2) * self.coeffs[i+1]  # [2*c[1], 3*c[2], ...]
        self.pprime_lo = self.coeffs[0] + polyval(dcoeffs, self.ncoeffs-1,
                                                  self.l_lo)
        self.pprime_hi = self.coeffs[0] + polyval(dcoeffs, self.ncoeffs-1,
                                                  self.l_hi)

    def __call__(self, double[:] wave):
        cdef:
            double l
            int i, n
            np.ndarray[np.float64_t, ndim=1] out

        n = wave.shape[0]
        out = np.empty(n, dtype=np.float64)

        for i in range(n):
            l = (wave[i] - SALT2CL_B) / SALT2CL_V_MINUS_B

            # Blue side
            if l < self.l_lo:
                out[i] = self.p_lo + self.pprime_lo * (l - self.l_lo)

            # in between
            elif l <= self.l_hi:
                out[i] = polyval(self.coeffs, self.ncoeffs, l)

            # red side
            else:
                out[i] = self.p_hi + self.pprime_hi * (l - self.l_hi)

            out[i] = -out[i]

        return out

    def __getnewargs__(self):
        """Return arguments to pass to constructor (to support pickling)."""

        # Note: an alternative to this would be moving the current __cinit__
        # contents to __init__, then defining __getstate__ and __setstate__
        # to copy the entire struct contents to a bytearray object and back.
        # This would avoid the need to do the initialization again, but
        # seems quite fiddly because the size to copy is unclear:
        # it depends on the struct layout.

        # reconstruct input wavelengths
        wave_lo = self.l_lo * SALT2CL_V_MINUS_B + SALT2CL_B
        wave_hi = self.l_hi * SALT2CL_V_MINUS_B + SALT2CL_B
        coeffs = [self.coeffs[i+1] for i in range(self.ncoeffs)]

        return (wave_lo, wave_hi), coeffs
