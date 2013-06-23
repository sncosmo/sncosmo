
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
--------

SNCosmo is a python library for supernova cosmology. It aims to make
dealing with various supernova models as easy as possible, while still
being completely extensible. It is built on NumPy, SciPy and AstroPy.

Usage
,,,,,

Get a built-in model and set its parameters:

    >>> import sncosmo
    >>> model = sncosmo.get_model('salt2')  # Get a built-in model
    >>> model.set(c=0.05, x1=0.5, mabs=-19.3, z=1.0)  # set some parameters

Calculate synthetic photometry at 3 phases:

    >>> model.bandflux('sdssr', [-25., 0., 40.])  # in ergs/s/cm^2
    >>> model.bandmag('sdssr', 'ab', [-25., 0., 40.])  # in AB mags

Get a spectrum at a given phase:

    >>> model.disp()  # observer-frame wavelength values in Angstroms.
    >>> model.flux(0.) # flux values in erg/s/cm^2/A at phase=0.

Key Features
,,,,,,,,,,,,

- **Built-ins:** There are many other built-in supernova models
  accessible by name (both Type Ia and core-collapse), such as Hsiao,
  Nugent, and models from PSNID. Common bandpasses and magnitude
  systems are also built-in and available by name (as demonstrated
  above).

- **Object-oriented and extensible:** New models, bandpasses, and
  magnitude systems can be defined, using an object-oriented interface.

- **Fitting and typing:** You can fit model parameters
  to data, or perform Bayesian model selection (photometric typing) by
  comparing multiple models to data.

Documentation
-------------

.. toctree::
   :maxdepth: 1

   install
   models
   bandpasses
   magsystems
   registry
   fitting
   typing

.. toctree::
   :maxdepth: 1

   reference

