from __future__ import absolute_import

import os
import sys
import math
import warnings
import socket

import numpy as np
from scipy import integrate, optimize


def format_value(value, error=None, latex=False):
    """Return a string representing value and uncertainty.

    If latex=True, use '\pm' and '\times'.
    """

    if latex:
        pm = '\pm'
        suffix_templ = ' \\times 10^{{{0:d}}}'
    else:
        pm = '+/-'
        suffix_templ = ' x 10^{0:d}'

    # First significant digit
    absval = abs(value)
    if absval == 0.:
        first = 0
    else:
        first = int(math.floor(math.log10(absval)))

    if error is None or error == 0.:
        last = first - 6  # Pretend there are 7 significant figures.
    else:
        last = int(math.floor(math.log10(error)))  # last significant digit

    # use exponential notation if
    # value > 1000 and error > 1000 or value < 0.01
    if (first > 2 and last > 2) or first < -2:
        value /= 10**first
        if error is not None:
            error /= 10**first
        p = max(0, first - last + 1)
        suffix = suffix_templ.format(first)
    else:
        p = max(0, -last + 1)
        suffix = ''

    if error is None:
        prefix = ('{0:.' + str(p) + 'f}').format(value)
    else:
        prefix = (('{0:.' + str(p) + 'f} {1:s} {2:.' + str(p) + 'f}')
                  .format(value, pm, error))
        if suffix != '':
            prefix = '({0})'.format(prefix)

    return prefix + suffix


class Result(dict):
    """Represents an optimization result.

    Notes
    -----
    This is a cut and paste from scipy, normally imported with `from
    scipy.optimize import Result`. However, it isn't available in
    scipy 0.9 (or possibly 0.10), so it is included here.
    Since this class is essentially a subclass of dict with attribute
    accessors, one can see which attributes are available using the
    `keys()` method.

    Deprecated attributes can be added via, e.g.:

        >>> res = Result(a=1, b=2)
        >>> res.__dict__['deprecated']['c'] = (2, "Use b instead")

    """

    # only necessary for deprecation functionality
    def __init__(self, *args, **kwargs):
        self.__dict__['deprecated'] = {}
        dict.__init__(self, *args, **kwargs)

    # only necessary for deprecation functionality
    def __getitem__(self, name):
        try:
            return dict.__getitem__(self, name)
        except:
            val, msg = self.__dict__['deprecated'][name]
            warnings.warn(msg)
            return val

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"


def _integral_diff(x, pdf, a, q):
    """Return difference between q and the integral of the function `pdf`
    between a and x. This is used for solving for the ppf."""
    return integrate.quad(pdf, a, x)[0] - q


def ppf(pdf, x, a, b):
    """Percent-point function (inverse cdf), given the probability
    distribution function pdf and limits a, b.

    Parameters
    ----------
    pdf : callable
        Probability distribution function
    x : array_like
        Points at which to evaluate the ppf
    a, b : float
        Limits (can be -np.inf, np.inf, assuming pdf has finite integral).
    """

    FACTOR = 10.

    if not b > a:
        raise ValueError('b must be greater than a')

    # integral of pdf between a and b
    tot = integrate.quad(pdf, a, b)[0]

    # initialize result array
    x = np.asarray(x)
    shape = x.shape
    x = np.ravel(x)
    result = np.zeros(len(x))

    for i in range(len(x)):
        cumsum = x[i] * tot  # target cumulative sum
        left = a
        right = b

        # Need finite limits for the solver.
        # For inifinite upper or lower limits, find finite limits such that
        # cdf(left) < cumsum < cdf(right).
        if left == -np.inf:
            left = -FACTOR
            while integrate.quad(pdf, a, left)[0] > cumsum:
                right = left
                left *= FACTOR
        if right == np.inf:
            right = FACTOR
            while integrate.quad(pdf, a, right)[0] < cumsum:
                left = right
                right *= FACTOR

        result[i] = optimize.brentq(_integral_diff, left, right,
                                    args=(pdf, a, cumsum))

    return result.reshape(shape)


class Interp1D(object):
    def __init__(self, xmin, xmax, y):
        self._xmin = xmin
        self._xmax = xmax
        self._n = len(y)
        self._xstep = (xmax - xmin) / (self._n - 1)
        self._y = y

    def __call__(self, x):
        """works only in range [xmin, xmax)"""
        nsteps = (x - self._xmin) / self._xstep
        i = int(nsteps)
        w = nsteps - i
        return (1.-w) * self._y[i] + w * self._y[i+1]


def _download_file(remote_url, target):
    """
    Accepts a URL, downloads the file to a given open file object.

    This is a modified version of astropy.utils.data.download_file that
    downloads to an open file object instead of a cache directory.
    """

    from contextlib import closing
    from six.moves.urllib.request import urlopen, Request
    from six.moves.urllib.error import URLError
    from astropy.utils.console import ProgressBarOrSpinner
    from astropy.utils.data import conf

    timeout = conf.remote_timeout

    try:
        # Pretend to be a web browser (IE 6.0). Some servers that we download
        # from forbid access from programs.
        headers = {'User-Agent': 'Mozilla/5.0',
                   'Accept': ('text/html,application/xhtml+xml,'
                              'application/xml;q=0.9,*/*;q=0.8')}
        req = Request(remote_url, headers=headers)
        with closing(urlopen(req, timeout=timeout)) as remote:

            # get size of remote if available (for use in progress bar)
            info = remote.info()
            size = None
            if 'Content-Length' in info:
                try:
                    size = int(info['Content-Length'])
                except ValueError:
                    pass

            dlmsg = "Downloading {0}".format(remote_url)
            with ProgressBarOrSpinner(size, dlmsg) as p:
                bytes_read = 0
                block = remote.read(conf.download_block_size)
                while block:
                    target.write(block)
                    bytes_read += len(block)
                    p.update(bytes_read)
                    block = remote.read(conf.download_block_size)

    # Append a more informative error message to URLErrors.
    except URLError as e:
        append_msg = (hasattr(e, 'reason') and hasattr(e.reason, 'errno') and
                      e.reason.errno == 8)
        if append_msg:
            msg = "{0}. requested URL: {1}".format(e.reason.strerror,
                                                   remote_url)
            e.reason.strerror = msg
            e.reason.args = (e.reason.errno, msg)
        raise e

    # This isn't supposed to happen, but occasionally a socket.timeout gets
    # through.  It's supposed to be caught in `urrlib2` and raised in this
    # way, but for some reason in mysterious circumstances it doesn't. So
    # we'll just re-raise it here instead.
    except socket.timeout as e:
        raise URLError(e)


def download_file(remote_url, local_name):
    """
    Download a remote file to local path, unzipping if the URL ends in '.gz'.

    Parameters
    ----------
    remote_url : str
        The URL of the file to download

    local_name : str
        Absolute path filename of target file.

    Raises
    ------
    URLError (from urllib2 on PY2, urllib.request on PY3)
        Whenever there's a problem getting the remote file.
    """

    # ensure target directory exists
    dn = os.path.dirname(local_name)
    if not os.path.exists(dn):
        os.makedirs(dn)

    if remote_url.endswith(".gz"):
        import io
        from astropy.utils.compat import gzip

        buf = io.BytesIO()
        _download_file(remote_url, buf)
        buf.seek(0)
        f = gzip.GzipFile(fileobj=buf, mode='rb')

        with open(local_name, 'wb') as target:
            target.write(f.read())
        f.close()

    else:
        with open(local_name, 'wb') as target:
            _download_file(remote_url, target)


def download_dir(remote_url, dirname):
    """
    Download a remote tar file to a local directory.

    Parameters
    ----------
    remote_url : str
        The URL of the file to download

    dirname : str
        Directory in which to place contents of tarfile. Created if it
        doesn't exist.

    Raises
    ------
    URLError (from urllib2 on PY2, urllib.request on PY3)
        Whenever there's a problem getting the remote file.
    """

    import io
    import tarfile

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    mode = 'r:gz' if remote_url.endswith(".gz") else None

    # download file to buffer
    buf = io.BytesIO()
    _download_file(remote_url, buf)
    buf.seek(0)

    # create a tarfile with the buffer and extract
    tf = tarfile.open(fileobj=buf, mode=mode)
    tf.extractall(path=dirname)
    tf.close()
    buf.close()  # buf not closed when tf is closed.
