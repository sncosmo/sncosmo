***************************
Package Overview & Examples
***************************

Package Functionality
=====================

At the core of the library is a class (and subclasses) for
representing supernova models: A model represents how the
spectroscopic time series of a transient astronomical source appears
to an observer. The spectroscopic times series can vary as a function
of any number of parameters (e.g., color of the source, redshift of
the source). Simulation, fitting and typing are built on this core
functionality.

Key Features
------------

- **Built-ins:** There are many built-in supernova models accessible
  by name (both Type Ia and core-collapse), such as Hsiao, Nugent, and
  models from PSNID (Sako et al 2011). Common bandpasses and magnitude
  systems are also built-in and available by name.

- **Object-oriented and extensible:** New models, bandpasses, and
  magnitude systems can be defined, using an object-oriented interface.

- **Fast:** Fully NumPy-ified and profiled. Generating
  synthetic photometry for 100 observations spread between four
  bandpasses takes on the order of 2 milliseconds (depends on model
  and bandpass sampling).

Examples
--------

**Model synthetic photometry**

::

    >>> import sncosmo
    >>> model = sncosmo.get_model('salt2')
    >>> model.set(c=0.05, x1=0.5, mabs=-19.3, z=1.0, t0=55100.)
    >>> model.bandmag('sdssr', 'ab', times=[55075., 55100., 55140.])
    array([ 27.1180182 ,  25.68243714,  28.28456537])

*See more in* :doc:`models` *and* :doc:`builtins/models`


**Model spectra**

::

    >>> from matplotlib import plot as plt
    >>> wl, flux = model.disp(), model.flux(time=55110.)
    >>> plt.plot(wl, flux)

.. image:: _static/example_spectrum.png  
   :width: 350px
   :height: 204px

*See more in* :doc:`models`

**Read and Write Photometric Data**

::

   >>> meta, data = sncosmo.read_lc('mydata.dat', fmt='csv')
   >>> sncosmo.write_lc(data, 'mydata.json', meta=meta, fmt='json')
   >>> sncosmo.write_lc(data, 'mydata.dat', meta=meta, fmt='salt2')
   >>> sncosmo.write_lc(data, 'mydata.fits', meta=meta, fmt='fits')

*See more in* :doc:`photdata`

**Fitting Light Curves**

::

    >>> res = sncosmo.fit_lc(data, model, ['x1','c','z','mabs','t0'],
    ...                      bounds={'z': (0.3, 0.7)})
    >>> res.params['x1'], res.errors['x1']
    (0.14702167554607398, 0.033596743599762925)

*See more in* :doc:`fitting`

**Quick plots**

::

    >>> model.set(**res.params)  # set parameters to best-fit values
    >>> sncosmo.plot_lc(data, model)

.. image:: _static/example_lc.png
   :width: 400px
   :height: 300px

*See more under "Plotting"* in the :doc:`reference`

**Photometric Typing**

::

    >>> typer = sncosmo.PhotoTyper()
    >>> sn1a_parlims = {'z': (0.01, 1.2), 'c':(-0.4, 0.6), 's': (0.7, 1.3),
    ...                 'mabs':(-18., -20.)}
    >>> sncc_parlims = {'z': (0.01, 1.1), 'c':(0., 0.6), 'mabs':(-17., -19.)}
    >>> typer.add_model('hsiao', 'SN Ia', sn1a_parlims)
    >>> typer.add_model('s11-2004hx', 'SN IIL', sncc_parlims)
    >>> types, models = typer.classify(data)
    >>> types['SN Ia']['p']
    1.0
    >>> models['hsiao']['p'], models['hsiao']['perr']
    (1.0, 0.0)

*See more in* :doc:`typing`


Package Scope
=============

We will consider for inclusion any functionality that is relevant to
supernova cosmology and of general use (that is, not specific to a
single survey or instrument). For example, cosmological fits may
eventually be included. The goal is to create a collection of Python
tools for use by, and developed by, the entire SN cosmology community.

Relation to core ``astropy`` package
------------------------------------

The package currently contains some functionality that is planned for
inclusion in the core ``astropy`` package or affiliated packages. As
this functionality is implemented in the core, we will transition to
using that functionality, provided that there are not significant
performance issues. Also, some general functionality implemented in
this package might propagate upward into the core ``astropy`` package.


Relation to other SN cosmology codes
====================================

There are several other publicly available software packages for
supernova cosmology. These include (but are not limited to) `snfit`_
(SALT fitter), `SNANA`_ and `SNooPy`_ (or snpy).

* `snfit`_ and `SNANA`_ both provide functionality overlapping with
  this package to some extent. The key difference is that these
  packages provide several (or many) executable applications, but do
  not provide an API for writing new programs building on the
  functionality they provide. This package, in contrast, provides no
  executables; instead it is a *library* of functions and classes
  designed to provide the building blocks commonly used in many
  aspects of SN analyses.

* `SNooPy`_ (or snpy) is also a Python library for SN analysis, but
  with a (mostly) different feature set. The current maintenance and
  development status of the package is unclear.


.. _`snfit`: http://supernovae.in2p3.fr/~guy/salt/index.html
.. _`SNANA`: http://sdssdp62.fnal.gov/sdsssn/SNANA-PUBLIC/
.. _`SNooPy`: http://csp.obs.carnegiescience.edu/data/snpy


The name "sncosmo"
==================

A natural choice, "snpy", was already taken (`SNooPy`_) so I tried to
be a little more descriptive. The package is really specific to
supernova *cosmology*, as it doesn't cover other types of supernova
science (radiative transfer simulations for instance).  Hence
"sncosmo".

Contributing to SNCosmo
=======================

.. _`issue tracker`: http://github.com/sncosmo/sncosmo/issues
.. _`contributing`: http://astropy.readthedocs.org/en/latest/development/workflow/index.html

This package is being actively developed. Bug reports, comments, and
help with development are very welcome.

Report issues
-------------

Even if you don't have time to contribute code or documentation,
please make sure you report any issues with the package or
documentation to the `issue tracker`_!

Contribute code
---------------

If you are interested in contributing fixes, code or documentation to
SNCosmo, take a look at the documentation pages on `contributing`_ to
Astropy. The idea is that the workflow for SNCosmo is very similar,
but with http://github.com/sncosmo/sncosmo functioning as the central
"blessed" repository in place of http://github.com/astropy/astropy
. You can either send a patch, or (preferably) work on a fork of
SNCosmo and submit the changes via a pull request. For big changes, it
is better to discuss your plans first before writing a lot of code.


Version History
===============

.. toctree::
   :maxdepth: 1

   whatsnew/0.4
   whatsnew/0.3
   whatsnew/0.2

.. note::
   For the time being, I am proceeding with minor version releases,
   which both add functionality and fix bugs. That is, there will not
   be independent bug-fix releases (e.g., v0.2.1) for these versions.

   This package uses `Semantic Versioning`_.

.. _`Semantic Versioning`: http:\\semver.org
