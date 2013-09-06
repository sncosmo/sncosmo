****************
Package Overview
****************

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
  bandpasses takes on the order of 1 millisecond (depends on model
  and bandpass sampling).

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


Current Stability
=================

The basic features of the models, bandpasses and magnitude systems can
be considered "fairly" stable (but no promises until v1.0).  The
fitting and typing functionalities are more experimental and the API
may change as it gets more real-world testing.

The name "sncosmo"
==================

A natural choice, "snpy", was already taken (`SNooPy`_) so I tried to
be a little more descriptive. The package is really specific to
supernova *cosmology*, as it doesn't cover other types of supernova
science (radiative transfer simulations for instance).  Hence
"sncosmo".

Contributing to SNCosmo
=======================

.. _`issue tracker`: http://github.com/kbarbary/sncosmo/issues
.. _`contributing`: http://astropy.readthedocs.org/en/latest/development/workflow/index.html

Anyone is welcome to contribute to SNCosmo.

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
but with http://github.com/kbarbary/sncosmo functioning as the central
"blessed" repository in place of http://github.com/astropy/astropy
. You can either send a patch, or (preferably) work on a fork of
SNCosmo and submit the changes via a pull request. For big changes, it
is better to discuss your plans first before writing a lot of code.


Version History
===============

.. toctree::
   :maxdepth: 1

   whatsnew/0.3
   whatsnew/0.2

.. note::
   For the time being, I am proceeding with minor version releases,
   which both add functionality and fix bugs. That is, there will not
   be independent bug-fix releases (e.g., v0.2.1) for these versions.

.. note::
   This package uses `Semantic Versioning`_, with the exception that the
   bugfix component of the version string is dropped for bugfix version 0.
   For example, v0.2 instead of v0.2.0.

.. _`Semantic Versioning`: http:\\semver.org
