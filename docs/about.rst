*************
About SNCosmo
*************

Package Features
================

- **SN models:** Synthesize supernova spectra and photometry from SN models.

- **Fitting and sampling:** Functions for fitting and sampling SN
  model parameters given photometric light curve data.

- **Dust laws:** Fast implementations of several commonly used
  extinction laws; can be used to construct SN models including dust.

- **I/O:** Convenience functions for reading and writing peculiar data formats
  used in other packages and getting dust values from SFD (1998) maps.

- **Built-in supernova models** such as the Hsiao, Nugent, PSNID,
  SNANA and Whalen models, as well as a variety of built-in bandpasses
  and magnitude systems.

- **Extensible:** New models, bandpasses, and
  magnitude systems can be defined, using an object-oriented interface.

- **Fast:** Fully NumPy-ified and profiled. Generating
  synthetic photometry for 100 observations spread between four
  bandpasses takes on the order of 2 milliseconds (depends on model
  and bandpass sampling).


Relation to other SN cosmology software
=======================================

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


The name SNCosmo
================

A natural choice, "snpy", was already taken (`SNooPy`_) so I tried to
be a little more descriptive. The package is really specific to
supernova *cosmology*, as it doesn't cover other types of supernova
science (radiative transfer simulations for instance).  Hence
"sncosmo".

Version History
===============

.. toctree::
   :maxdepth: 1

   whatsnew/1.0
   whatsnew/0.4
   whatsnew/0.3
   whatsnew/0.2

This package uses `Semantic Versioning`_.

.. _`Semantic Versioning`: http://semver.org

Contributors
============

* Kyle Barbary
* Rahul Biswas
* Steve Rodney
* Caroline Sofiatti
* Tom Barclay
* Rollin C. Thomas
* Danny Goldstein
