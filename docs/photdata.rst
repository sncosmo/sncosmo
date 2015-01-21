****************
Photometric Data
****************

Introduction
============

In order to fit a supernova model to photometric data, we must first
have a standard way to represent the photometric data so that a
fitting function can interpret it. We've chosen to use an astropy
`~astropy.table.Table` for this standard representation: each
photometric data point is a row in the table. To see what such a table
looks like, you can load an example with the following function::


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

Additionally, metadata about the photometric data can be stored with
the table: ``data.meta`` is an `OrderedDict` of the metadata. Additional
columns in the table can exist, but the above columns (with aliases for
the names defined in the next section) are necessary.


Column names and units
======================

The example data table above has the minimum six columns necessary for
sncosmo's light curve fitting and plotting functions to interpret the
data. (There's no harm in having more columns for other supplementary
data.) There is some flexibility in the column names that sncosmo
recognizes. For example, you can use ``filter`` for a column name
instead of ``band``:

.. automodule:: sncosmo.photdata

The units of the flux and flux uncertainty are given by the zeropoint
system, with the zeropoint itself serving as a scaling factor: For
example, if the zeropoint is ``25.0`` and the zeropoint system is
``'vega'``, a flux of 1.0 corresponds to ``10**(-25/2.5)`` times the
integrated flux of Vega in the given bandpass.


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
| salt2       | SALT2 new-style data files  | Mostly untested.              |
+-------------+-----------------------------+-------------------------------+
| salt2-old   | SALT2 old-style data files  | Mostly untested.              |
+-------------+-----------------------------+-------------------------------+


Manipulating data tables
========================

Rename a column::

    >>> data.rename_column('oldname', 'newname')

Add a column::

    >>> data['zp'] = 26.

Add a constant value to all the entries in a given column::

    >>> data['zp'] += 0.03

See the documentation on astropy tables for more information.
