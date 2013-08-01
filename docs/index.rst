
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

SNCosmo aims to make high-level supernova cosmology analysis as easy
as possible, while still being completely extensible. It is built on
NumPy, SciPy and AstroPy.

**Simulation & synthetic photometry** ::

    >>> import sncosmo
    >>> model = sncosmo.get_model('salt2')
    >>> model.set(c=0.05, x1=0.5, mabs=-19.3, z=1.0)
    >>> model.bandmag('sdssr', 'ab', [-25., 0., 40.])

**Model spectra** ::

    >>> wl, flux = model.disp(), model.flux(0.)

.. toctree::
   :hidden:
   :maxdepth: 1

   overview
   install
   models
   bandpasses
   magsystems
   photometric_data
   plotting
   fitting
   typing
   registry
   reference
   development
