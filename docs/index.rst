
.. raw:: html

    <style media="screen" type="text/css">
      h1 { display:none; }
    </style>

*******
SNCosmo
*******

.. image:: _static/sncosmo_banner_96.png
    :width: 409px
    :height: 96px

Overview
========

SNCosmo is a python library for supernova cosmology. It aims to make
dealing with various supernova models as easy as possible, while still
being completely extensible. It is built on NumPy, SciPy and AstroPy.

At the core of the library is a class (and subclasses) for
representing supernova models: A model is essentially a spectroscopic
time series that may vary as a function of one or more
parameters. Simulation, fitting and typing can all be naturally built
on this core functionality.

Example Usage
-------------

Get a built-in model and set its parameters:

    >>> import sncosmo
    >>> model = sncosmo.get_model('salt2')  # Get a built-in model
    >>> model.set(c=0.05, x1=0.5, mabs=-19.3, z=1.0)  # set some parameters

Different types of models can have different parameters:

    >>> model = sncosmo.get_model('hsiao')
    >>> model.set(c=0.05, s=1.1, mabs=-19.3, z=1.0)

Calculate synthetic photometry at 3 phases:

    >>> model.bandflux('sdssr', [-25., 0., 40.])  # in photons/s/cm^2
    >>> model.bandmag('sdssr', 'ab', [-25., 0., 40.])  # in AB mags

Get a spectrum at a given phase:

    >>> model.disp()  # observer-frame wavelength values in Angstroms.
    >>> model.flux(0.) # flux values in erg/s/cm^2/A at phase=0.

Key Features
------------

- **Built-ins:** There are many other built-in supernova models
  accessible by name (both Type Ia and core-collapse), such as Hsiao,
  Nugent, and models from PSNID. Common bandpasses and magnitude
  systems are also built-in and available by name (as demonstrated
  above).

- **Object-oriented and extensible:** New models, bandpasses, and
  magnitude systems can be defined, using an object-oriented interface.

- **Fast:** Fully NumPy-ified and profiled. Generating
  synthetic photometry for 100 observations spread between four
  bandpasses takes on the order of 1 millisecond (depends on model
  and bandpass sampling).

Stability
---------

The basic features of the models, bandpasses and magnitude systems
(documented below) can be considered "fairly" stable (but no
guarantees).  The fitting and typing functionalities are more
experimental and the API may change as it gets more real-world
testing.

Development
-----------

Bug reports, comments, and help with development are very welcome.
Source code and issue tracking is hosted on github:
https://github.com/kbarbary/sncosmo

.. toctree::
   :maxdepth: 1
   :hidden:

   future_development

:doc:`future_development`


User Documentation
==================

.. toctree::
   :maxdepth: 1

   install
   whatsnew/0.2

Core
----

.. toctree::
   :maxdepth: 1

   models
   bandpasses
   magsystems
   registry

Experimental Features
---------------------

.. toctree::
   :maxdepth: 1

   photometric_data
   fitting
   typing

Reference / API
===============

Built-ins
---------

.. toctree::
   :maxdepth: 1

   builtins/models
   builtins/bandpasses
   builtins/magsystems

.. currentmodule:: sncosmo

Functions
---------

.. autosummary::
   :toctree: _generated

   get_model
   get_bandpass
   get_magsystem
   extinction_ccm
   registry.register_loader
   registry.register
   registry.retrieve

Classes
-------

.. autosummary::
   :toctree: _generated

   TimeSeriesModel
   StretchModel
   SALT2Model
   Bandpass
   SpectralMagSystem
   ABMagSystem

Experimental Functions
----------------------

.. autosummary::
   :toctree: _generated

   fit_model
   readlc
   writelc
   plotlc

Experimental Classes
--------------------

.. autosummary::
   :toctree: _generated

   PhotoTyper
