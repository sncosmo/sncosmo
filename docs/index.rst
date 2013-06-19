
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


Features
--------

SNCosmo is a python library for supernova cosmology. Some features:

- Many built-in supernova models (Type Ia and core-collapse), such as Hsiao,
  Nugent, SALT2, and models from PSNID.
- Generate synthetic photometry from models for any combination of parameters,
  including extinction, redshift, absolute magnitude.
- Extract spectrum from a model for any time(s) and wavelength values.
- Extensible to new models, user-defined magnitude systems, and
  user-defined bandpasses.
- Fit a model to photometric data.
- Photometric typing.
- Simple light curve plotting.
- Read and write various light curve file formats, such as SALT2, SNANA.

See the :doc:`overview`.

Documentation
-------------

.. toctree::
   :hidden:

   overview

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

