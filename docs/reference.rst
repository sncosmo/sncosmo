
Reference/API
=============

.. currentmodule:: snsim

Classes
-------

These classes contain the core functionality of the library. 

.. autosummary::
   :toctree: _generated/

   Bandpass
   Spectrum
   models.Transient
   models.TimeSeries
   models.SALT2
   Survey


Factory Functions
-----------------

Read built-in data (from the path given by the user's ``$SNSIM_DATA``
environment variable) and create a class instance based on that data.
For a list of available built-ins, see the function documentation.

.. autosummary::
   :toctree: _generated/

   bandpass
   spectrum
   model


I/O Utilities (:mod:`snsim.io`)
-------------------------------

Utilities for reading and writing file formats not convered by :mod:`astropy`.

.. autosummary::
   :toctree: _generated/

   io.read_griddata_txt
   io.read_simlib
   io.salt2.read
   io.salt2.write
   io.salt2.readdir
   io.salt2.writedir

General Utilities (:mod:`snsim.utils`)
--------------------------------------

Utilities with functionality not specific to the :mod:`snsim` package.

.. autosummary::
   :toctree: _generated/

   utils.GridData
