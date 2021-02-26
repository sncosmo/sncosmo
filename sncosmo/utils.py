import codecs
import math
import os
import socket
import warnings
from collections import OrderedDict

import numpy as np
from scipy import integrate, optimize


def dict_to_array(d):
    """Convert a dictionary of lists (or single values) to a structured
    numpy.ndarray."""

    # Convert all lists/values to 1-d arrays, in order to let numpy
    # figure out the necessary size of the string arrays.
    new_d = OrderedDict()
    for key in d:
        new_d[key] = np.atleast_1d(d[key])

    # Determine dtype of output array.
    dtype = [(key, arr.dtype)
             for key, arr in new_d.items()]

    # Initialize ndarray and then fill it.
    col_len = max([len(v) for v in new_d.values()])
    result = np.empty(col_len, dtype=dtype)
    for key in new_d:
        result[key] = new_d[key]

    return result


def format_value(value, error=None, latex=False):
    """Return a string representing value and uncertainty.

    If latex=True, use '\\pm' and '\\times'.
    """

    if latex:
        pm = '\\pm'
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
        except KeyError:
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
    from urllib.request import urlopen, Request
    from urllib.error import URLError, HTTPError
    from astropy.utils.console import ProgressBarOrSpinner
    from . import conf

    timeout = conf.remote_timeout
    download_block_size = 32768
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
                block = remote.read(download_block_size)
                while block:
                    target.write(block)
                    bytes_read += len(block)
                    p.update(bytes_read)
                    block = remote.read(download_block_size)

    # Append a more informative error message to HTTPErrors, URLErrors.
    except HTTPError as e:
        e.msg = "{}. requested URL: {!r}".format(e.msg, remote_url)
        raise
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
        # add the requested URL to the message (normally just 'timed out')
        e.args = ('requested URL {!r} timed out'.format(remote_url),)
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
    URLError
        Whenever there's a problem getting the remote file.
    """

    # ensure target directory exists
    dn = os.path.dirname(local_name)
    if not os.path.exists(dn):
        os.makedirs(dn)

    if remote_url.endswith(".gz"):
        import io
        import gzip

        buf = io.BytesIO()
        _download_file(remote_url, buf)
        buf.seek(0)
        f = gzip.GzipFile(fileobj=buf, mode='rb')

        with open(local_name, 'wb') as target:
            target.write(f.read())
        f.close()

    else:
        try:
            with open(local_name, 'wb') as target:
                _download_file(remote_url, target)
        except:  # noqa
            # in case of error downloading, remove file.
            if os.path.exists(local_name):
                os.remove(local_name)
            raise


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


class DataMirror(object):
    """Lazy fetcher for remote data.

    When asked for local absolute path to a file or directory, DataMirror
    checks if the file or directory exists locally and, if so, returns it.

    If it doesn't exist, it first determines where to get it from.
    It first downloads the file ``{remote_root}/redirects.json`` and checks
    it for a redirect from ``{relative_path}`` to a full URL. If no redirect
    exists, it uses ``{remote_root}/{relative_path}`` as the URL.

    It downloads then downloads the URL to ``{rootdir}/{relative_path}``.

    For directories, ``.tar.gz`` is appended to the
    ``{relative_path}`` before the above is done and then the
    directory is unpacked locally.

    Parameters
    ----------
    rootdir : str or callable

        The local root directory, or a callable that returns the local root
        directory given no parameters. (The result of the call is cached.)
        Using a callable allows one to customize the discovery of the root
        directory (e.g., from a config file), and to defer that discovery
        until it is needed.

    remote_root : str
        Root URL of the remote server.
    """

    def __init__(self, rootdir, remote_root):
        if not remote_root.endswith('/'):
            remote_root = remote_root + '/'

        self._checked_rootdir = None
        self._rootdir = rootdir
        self._remote_root = remote_root

        self._redirects = None

    def rootdir(self):
        """Return the path to the local data directory, ensuring that it
        exists"""

        if self._checked_rootdir is None:

            # If the supplied value is a string, use it. Otherwise
            # assume it is a callable that returns a string)
            rootdir = (self._rootdir
                       if isinstance(self._rootdir, str)
                       else self._rootdir())

            # Check existance
            if not os.path.isdir(rootdir):
                raise Exception("data directory {!r} not an existing "
                                "directory".format(rootdir))

            # Cache value for future calls
            self._checked_rootdir = rootdir

        return self._checked_rootdir

    def _fetch_redirects(self):
        from urllib.request import urlopen
        import json

        f = urlopen(self._remote_root + "redirects.json")
        reader = codecs.getreader("utf-8")
        self._redirects = json.load(reader(f))
        f.close()

    def _get_url(self, remote_relpath):
        if self._redirects is None:
            self._fetch_redirects()

        if remote_relpath in self._redirects:
            return self._redirects[remote_relpath]
        else:
            return self._remote_root + remote_relpath

    def abspath(self, relpath, isdir=False):
        """Return absolute path to file or directory, ensuring that it exists.

        If ``isdir``, look for ``{relpath}.tar.gz`` on the remote server and
        unpackage it.

        Otherwise, just look for ``{relpath}``. If redirect points to a gz, it
        will be uncompressed."""

        abspath = os.path.join(self.rootdir(), relpath)

        if not os.path.exists(abspath):
            if isdir:
                url = self._get_url(relpath + ".tar.gz")

                # Download and unpack a directory.
                download_dir(url, os.path.dirname(abspath))

                # ensure that tarfile unpacked into the expected directory
                if not os.path.exists(abspath):
                    raise RuntimeError("Tarfile not unpacked into expected "
                                       "subdirectory. Please file an issue.")
            else:
                url = self._get_url(relpath)
                download_file(url, abspath)

        return abspath


def alias_map(aliased, aliases, required=()):
    """For each key in ``aliases``, find the item in ``aliased`` matching
    exactly one of the corresponding items in ``aliases``.

    Parameters
    ----------
    aliased : list of str
        Input keys, will be values in output map.
    aliases : dict of sets
        Dictionary where keys are "canonical name" and values are sets of
        possible aliases.
    required : list_like
        Keys in ``aliases`` that are considered required. An error is raised
        if no alias is found in ``aliased``.


    Returns
    -------

    Example::

        >>> aliases = {'a':set(['a', 'a_']), 'b':set(['b', 'b_'])}
        >>> alias_map(['A', 'B_', 'foo'], aliases)
        {'a': 'A', 'b': 'B_'}



    """
    lowered_to_orig = {key.lower(): key for key in aliased}
    lowered = set(lowered_to_orig.keys())
    mapping = {}
    for key, key_aliases in aliases.items():
        common = lowered & key_aliases
        if len(common) == 1:
            mapping[key] = lowered_to_orig[common.pop()]

        elif len(common) == 0 and key in required:
            raise ValueError('no alias found for {!r} (possible '
                             'case-independent aliases: {})'.format(
                                 key,
                                 ', '.join(repr(ka) for ka in key_aliases)))
        elif len(common) > 1:
            raise ValueError('multiple aliases found for {!r}: {}'
                             .format(key, ', '.join(repr(a) for a in common)))

    return mapping


def integration_grid(low, high, target_spacing):
    """Divide the range between `start` and `stop` into uniform bins
    with spacing less than or equal to `target_spacing` and return the
    bin midpoints and the actual spacing."""

    range_diff = high - low
    spacing = range_diff / int(math.ceil(range_diff / target_spacing))
    grid = np.arange(low + 0.5 * spacing, high, spacing)

    return grid, spacing


warned = []  # global used in warn_once


def warn_once(name, depver, rmver, extra=None):
    global warned
    if name not in warned:
        msg = ("{} is deprecated in sncosmo {} "
               "and will be removed in sncosmo {}".format(name, depver, rmver))
        if extra is not None:
            msg += " " + extra
        warnings.warn(msg, stacklevel=2)
        warned.append(name)
