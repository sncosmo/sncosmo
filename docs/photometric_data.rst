****************
Photometric Data
****************

Quick start
===========

For an example of an acceptable photometric data representation, you can do::

    >>> meta, data = sncosmo.load_example_data()

``meta`` is an ordered dictionary and ``data`` is a `~numpy.ndarray`
representing a table of photometric observations. For a pretty-printed
version of this, you can do::

    >>> from astropy.table import Table
    >>> print Table(data)

Photometric Data: Structure
===========================

Usually, photometric data can naturally be represented as a table of
observations. Rather than imposing a new, unfamiliar format or class for
representing such data, the approach take in sncosmo is to accept
anything that "looks like a table" in the place where a set of
photometric data is required. This means that in the functions and
classes that accept photometric data as an argument, ``data`` can be
any of:

* A dictionary of columns. Each column is a list or `~numpy.ndarray`
  of matching lengths, or a single value. If a single value, it is
  interpreted as applying to all observations.
* A structured `~numpy.ndarray`

Required Columns
----------------

Whichever structure is used, the data must have certain recognizable
columns. For photometric data with Gaussian uncertainties in
flux-space (often a good approximation), the following quantities
uniquely specify a single observation:

* The time of the observation (e.g., MJD, etc)
* The bandpass in which the observation was taken
* The flux
* The uncertainty on the flux
* The zeropoint (and magnitude system) tying the flux to a physical system.

Therefore, ``data`` must have a column to represent each quantity. The
column names are somewhat flexible. The table below shows the
acceptable column names for ``data``. For example ``data`` must
contain exactly one of ``'date'``, ``'jd'``, ``'mjdobs'``, ``'mjd'``,
or ``'time'``.

.. automodule:: sncosmo.photometric_data

Reading and Writing photometric data from files
===============================================

SNCosmo strives to be agnostic with respect to file format. In
practice there are a plethora of different file formats, both standard
and non-standard, used to represent tables. Rather than picking a
single supported file format, or worse, creating yet another new
"standard", we choose to leave the file format mostly up to the user:
It is the user's job to read their data into either a `dict` or
`~numpy.ndarray`.

That said, SNCosmo does include a couple convenience functions for
reading and writing tables: `~sncosmo.read_lc` and
`~sncosmo.write_lc`.

The supported formats are listed below. If your preferred format is
not available, write your own parser or use a standard one from the
python universe.

+-------------+-----------------------------+-------------------------------+
| Format name | Description                 | Notes                         |
+=============+=============================+===============================+
| csv         | CSV-like, but with metadata | Not actually readable by      |
|             | lines marked by '@'         | standard CSV parsers :(       |
+-------------+-----------------------------+-------------------------------+
| fits        | Standard FITS               | Has poor performance and      |
|             |                             | storage size for small tables |
+-------------+-----------------------------+-------------------------------+
| json        | JavaScript Object Notation  | Good performance, but not as  |
|             |                             | human-readable as csv         |
+-------------+-----------------------------+-------------------------------+
| salt2       | SALT2 new-style data files  | Mostly untested.              |
+-------------+-----------------------------+-------------------------------+
| snana       | (Write-only) SNANA-like     | Utility and future support    |
|             | format                      | uncertain.                    |
+-------------+-----------------------------+-------------------------------+

To see what each format looks like, you can do, e.g.::

    >>> meta, data = sncosmo.load_example_data()
    >>> sncosmo.write_lc(data, fname='test.json', fmt='json')


A note on `astropy.table.Table`
===============================

In the future, we may move to using the `~astropy.table.Table`
provided in AstroPy, for representing tables and/or reading and
writing. However, the class's ``write`` method does not yet include a
human-readable format that includes metadata.
