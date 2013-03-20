#######
SNCosmo
#######


Description
-----------

SNCosmo is a python library for supernova cosmology. It provides a
class for empirical supernova models, allowing you to interact with
a variety of models in a uniform way. You can easily:

- extract model phase coverage
- extract model wavelength coverage
- extract model spectra
- extract model synthetic photometry
- redshift models
- apply extinction

Data for several models from the literature are "built-in", such as
the Hsiao, Nugent, and SALT2 models. See the
:ref:`list-of-built-in-models`. Inclusion of SALT2 model errors is
work in progress.

A Quick Example
---------------

Suppose you wish to simulate a SN Ia light curve at a
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
