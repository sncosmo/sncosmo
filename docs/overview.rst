****************
Package Overview
****************

Package Functionality
=====================

At the core of the library is a class (and subclasses) for
representing supernova models: A model is essentially a spectroscopic
time series that may vary as a function of one or more
parameters. Simulation, fitting and typing can all be naturally built
on this core functionality.

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
  with a (mostly) different feature set. SNooPy is aimed at both the
  Python programmer and the general user (you don't necessarily need
  to know Python to use it), whereas this package is (currently) only a
  library.


.. _`snfit`: http://supernovae.in2p3.fr/~guy/salt/index.html

.. _`SNANA`: http://sdssdp62.fnal.gov/sdsssn/SNANA-PUBLIC/

.. _`SNooPy`: http://csp.obs.carnegiescience.edu/data/snpy


Current Stability
=================

The basic features of the models, bandpasses and magnitude systems can
be considered "fairly" stable (but no promises until v1.0).  The
fitting and typing functionalities are more experimental and the API
may change as it gets more real-world testing.

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
