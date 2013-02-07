
Reference/API
=============

.. currentmodule:: sncosmo

Synthetic Photometry
--------------------

Currently this contains minimal functionality necessary to perform
basic operations on spectra, such as redshifting and synthetic
photometry. These operations are accomplished with the `Spectrum`
class. This functionality could eventually be replaced by `astropy`
if implemented there.

.. autosummary::
   :toctree: _generated/

   Bandpass
   Spectrum


Supernova Models (:mod:`sncosmo.models`)
----------------------------------------

Models of astrophysical transient sources. A model is anything where
the spectrum as a function of phase can be parameterized by an
arbitrary number of parameters. For example, the Hsiao and Nugent SN
templates are zero parameter models (not counting amplitude): there is
a single spectrum for a given phase. The SALT2 model has two
parameters (:math:`x1` and :math:`c`) that determine a unique spectrum
as a function of phase. The Hsiao and Nugent models can be described
using a single class (`models.TimeSeries`) whereas the SALT2 model
needs a separate subclass (`models.SALT2`). All models derive from a
common abstract base class (`models.Transient`).

.. autosummary::
   :toctree: _generated/

   models.Transient
   models.TimeSeries
   models.SALT2

.. inheritance-diagram:: models


Simulation (:mod:`sncosmo.sim`)
-------------------------------

Classes and functions for simulating SNe in surveys will live in this
subpackage. Functionality will be split between (at least) two steps:
"realizing" a set of SNe and simulating the observations of those SNe.

* *Realizing a set of SNe* Simulating the locations and properties of
  SNe given things like the intrinsic SN rate, distribution of SN
  properties, and correlations with properties and SN hosts. Here, the
  "properties of a SN" means the model and parameters that completely
  describe the full SED, independent of what observations might be
  made of that SN.

* *Simulating observations* Given properties of survey observations
  (simulated or actual) simulate the SN flux and flux error for an SN
  with some given properties (or set of SNe).



Factory Functions / Registry (:mod:`sncosmo.builtin`)
-----------------------------------------------------

Automatically create Bandpass, Spectrum and Model objects from
built-in data.  This is essentially a registry that ties a string
identifier (such as ``'salt2'`` to a class and data (such as the
`models.SALT2` class and some model data).

.. autosummary::
   :toctree: _generated/

   builtin.model
   builtin.bandpass
   builtin.spectrum


I/O Utilities (:mod:`sncosmo.io`)
---------------------------------

Utilities for reading and writing file formats not convered by :mod:`astropy`.

.. autosummary::
   :toctree: _generated/

   io.read_griddata_txt


General Utilities (:mod:`sncosmo.utils`)
----------------------------------------

Utilities with functionality not specific to this package.

.. autosummary::
   :toctree: _generated/

   utils.GridData
