#######
SNCosmo
#######

A python package for supernova cosmology based on :mod:`~astropy`.
This is currently a "working implementation", intended to demonstrate
the functionality and a possible API.

A quick example
---------------

There are a variety of built-in models for several types of SNe
(but mainly Type Ia). Built-in models are retrievable by name:

   >>> from sncosmo import Model
   >>> m = Model.from_name('hsiao')

Get the model spectrum at phase 5.3 days and calculate the AB magnitude in
the DES g bandpass (bandpasses and magnitude systems can also be specified
by name):

   >>> s = m.spectrum(5.3)
   >>> s.mag('desg', 'ab')


Documentation
-------------

.. toctree::
   :maxdepth: 1

   install
   models
   spectral
   utils
   registry
