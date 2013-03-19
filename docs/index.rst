#######
SNCosmo
#######


Description
-----------

SNCosmo is a python library for supernova cosmology. It provides a
class for empirical supernova models, allowing you to extract things
like the model phase coverage, wavelength coverage, spectra and
synthetic photometry using a common interface for several different
types of models.

A Quick Example
---------------

For example, suppose one wishes to simulate a SN Ia light curve at a
redshift of ``z=1`` using the SALT2 model, with color ``c=0.05``, x1
parameter of ``x1=0.5``, and a peak absolute AB magnitude of -19.3 in
the Bessell B band:

    >>> import sncosmo
    >>> model = sncosmo.get_model('salt2')
    >>> model.set(c=0.05, x1=0.5, absmag=(-19.3, 'bessellb', 'ab'), z=1.0)

To get the observer-frame magnitude in the SDSS *z* band at phases of
-10, 0, 10, 20 days (observer-frame):

    >>> model.mag([-10., 0., 10., 20.], 'sdssz', 'ab')
    array([ 24.26614271,  24.10994318,  24.27268101,  24.61381509])

For more example usage, see :doc:`models`.

Documentation
-------------

.. toctree::
   :maxdepth: 1

   install
   models
   bandpasses
   magsystems
   registry
   reference
