#######
SNCosmo
#######

SNCosmo is a python package for supernova cosmology based on the
`astropy`_ project. It provides a generalized class for empirical
supernova models and an interface for extracting things like their
spectra, wavelength coverage, and synthetic photometry.

You can initialize your own model from a set of model data, but a
number of models commonly used in the literature are available by name::

   >>> import sncosmo
   >>> model = sncosmo.get_model('nugent-sn1a')


Documentation
-------------

.. toctree::
   :maxdepth: 1

   install
   models
   spectral
   registry
   reference
