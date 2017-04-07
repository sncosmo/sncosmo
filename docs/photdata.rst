****************
Photometric Data
****************

Photometric data stored in AstroPy Table
========================================

In sncosmo, photometric data for a supernova is stored in an astropy
`~astropy.table.Table`: each row in the table is a photometric
observation.  The table must contain certain columns. To see what such
a table looks like, you can load an example with the following
function::


    >>> data = sncosmo.load_example_data()
    >>> print data
         time      band        flux          fluxerr      zp  zpsys
    ------------- ----- ----------------- -------------- ---- -----
          55070.0 sdssg    0.813499900062 0.651728140824 25.0    ab
    55072.0512821 sdssr  -0.0852238865812 0.651728140824 25.0    ab
    55074.1025641 sdssi -0.00681659003089 0.651728140824 25.0    ab
    55076.1538462 sdssz     2.23929135407 0.651728140824 25.0    ab
    55078.2051282 sdssg  -0.0308977349373 0.651728140824 25.0    ab
    55080.2564103 sdssr     2.35450321853 0.651728140824 25.0    ab
    ... etc ...

This example data table above has the minimum six columns necessary for
sncosmo's light curve fitting and plotting functions to interpret the
data. (There's no harm in having more columns for other supplementary
data.)

Additionally, metadata about the photometric data can be stored with
the table: ``data.meta`` is an `OrderedDict` of the metadata.

Including Covariance
====================

If your table contains a column ``'fluxcov'`` (or any similar name;
see below) it will be interpreted as covariance between the data
points and will be used *instead* of the ``'fluxerr'`` column when
calculating a :math:`\chi^2` value in fitting functions. For each row,
the ``'fluxcov'`` column should be a length *N* array, where *N* is the
number of rows in the table. In other, words, ``table['fluxcov']`` should
have shape ``(N, N)``, where other columns like ``table['time']`` have shape
``(N,)``.

As an example, let's add a ``'fluxcov'`` column to the example data
table above. ::

  >>> data['fluxcov'] = np.diag(data['fluxerr']**2)
  >>> len(data)
  40
  >>> data['fluxcov'].shape
  (40, 40)

  # diagonal elements are error squared:
  >>> data['fluxcov'][0, 0]
  0.45271884317377648

  >>> data['fluxerr'][0]
  0.67284384754100002

  # off diagonal elements are zero:
  >>> data['fluxcov'][0, 1]
  0.0

As is, this would be completely equivalent to just having the
``'fluxerr'`` column. But now we have the flexibility to represent
non-zero off-diagonal covariance.

.. note::

   When sub-selecting data from a table with covariance, be sure to
   use `sncosmo.select_data`. For example, rather than
   ``table[mask]``, use ``sncosmo.select_data(table, mask)``. This
   ensures that the covariance column is sliced appropriately! See the
   documentation for `~sncosmo.select_data` for details.

Flexible column names
=====================

What if you'd rather call the time column ``'date'``, or perhaps
``'mjd'``?  Good news! SNCosmo is flexible about the column names. For
each column, it accepts a variety of alias names:

.. automodule:: photdata_aliases_table

Note that each column must be present in some form or another, with no
repeats.  For example, you can have either a ``'flux'`` column or a
``'f'`` column, but not both.

The units of the flux and flux uncertainty are effectively given by
the zeropoint system, with the zeropoint itself serving as a scaling
factor: For example, if the zeropoint is ``25.0`` and the zeropoint
system is ``'vega'``, a flux of 1.0 corresponds to ``10**(-25/2.5)``
times the integrated flux of Vega in the given bandpass.


Reading and Writing photometric data from files
===============================================

SNCosmo strives to be agnostic with respect to file format. In
practice there are a plethora of different file formats, both standard
and non-standard, used to represent tables. Rather than picking a
single supported file format, or worse, creating yet another new
"standard", we choose to leave the file format mostly up to the user:
A user can use any file format as long as they can read their data
into an astropy `~astropy.table.Table`.

That said, SNCosmo does include a couple convenience functions for
reading and writing tables of photometric data: `sncosmo.read_lc` and
`sncosmo.write_lc`::

    >>> data = sncosmo.load_example_data()
    >>> sncosmo.write_lc(data, 'test.txt')

This creates an output file `test.txt` that looks like::

    @x1 0.5
    @c 0.2
    @z 0.5
    @x0 1.20482820761e-05
    @t0 55100.0
    time band flux fluxerr zp zpsys
    55070.0 sdssg 0.36351153597 0.672843847541 25.0 ab
    55072.0512821 sdssr -0.200801295864 0.672843847541 25.0 ab
    55074.1025641 sdssi 0.307494232981 0.672843847541 25.0 ab
    55076.1538462 sdssz 1.08776103656 0.672843847541 25.0 ab
    55078.2051282 sdssg -0.43667895645 0.672843847541 25.0 ab
    55080.2564103 sdssr 1.09780966779 0.672843847541 25.0 ab
    ... etc ...

Read the file back in::

    >>> data2 = sncosmo.read_lc('test.txt')

There are a few other available formats, which can be specified using
the ``format`` keyword::

    >>> data = sncosmo.read_lc('test.json', format='json')

The supported formats are listed below. If your preferred format is
not included, use a standard reader/writer from astropy or the Python universe.

+-------------+-----------------------------+-------------------------------+
| Format name | Description                 | Notes                         |
+=============+=============================+===============================+
| ascii       | ASCII with metadata         | Not readable by               |
| (default)   | lines marked by '@'         | standard ASCII table parsers  |
|             |                             | due to metadata lines.        |
+-------------+-----------------------------+-------------------------------+
| json        | JavaScript Object Notation  | Good performance, but not as  |
|             |                             | human-readable as ascii       |
+-------------+-----------------------------+-------------------------------+
| salt2       | SALT2 new-style data files  |                               |
+-------------+-----------------------------+-------------------------------+
| salt2-old   | SALT2 old-style data files  |                               |
+-------------+-----------------------------+-------------------------------+


Manipulating data tables
========================

Because photometric data tables are astropy Tables, they can be manipulated
any way that Tables can. Here's a few things you might want to do.

Rename a column::

    >>> data.rename_column('oldname', 'newname')

Add a column::

    >>> data['zp'] = 26.

Add a constant value to all the entries in a given column::

    >>> data['zp'] += 0.03

See the documentation on astropy tables for more information.
