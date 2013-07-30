****************
Photometric Data
****************

Quick guide
-----------

For an acceptable example of photometric data, you can do::

    >>> meta, data = sncosmo.load_example_data()

In this case, ``data`` is a `~numpy.ndarray` representing a table of
photometric observations.

Structure
---------

Usually, photometric data can naturally be represented as a table of
observations. Rather than imposing a new, unfamiliar format for
representing such data, the approach take in sncosmo is to accept
anything that "looks like a table" in the place where a set of
photometric data is required. This means that in the functions and
classes that accept photometric data as an argument, ``data`` can be
any of:

* A dictionary of columns (each column is a list or `~numpy.ndarray`
  of matching lengths.
* A structured `~numpy.ndarray`
* An astropy `~astropy.table.Table`

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
column names are somewhat flexible.

.. automodule:: sncosmo.photometric_data
