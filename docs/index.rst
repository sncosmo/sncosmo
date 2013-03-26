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

It's pretty fast
----------------

An emphasis is placed on computation speed so that the models can be fit to
data in a reasonable time. Synthetic photometry for 100 observations can be
evaluated in ~50 milliseconds:

    In [5]: ndata = 100
    In [6]: dates = np.zeros(ndata, dtype=np.float)
    In [7]: bands = np.array(ndata * ['sdssr'])
    In [8]: timeit model.bandflux(bands, dates)
    10 loops, best of 3: 49.4 ms per loop

This is without cython optimization.

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
