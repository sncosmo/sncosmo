*************
About SNCosmo
*************

Package Features
================

- **SN models:** Synthesize supernova spectra and photometry from SN
  models.

- **Fitting and sampling:** Functions for fitting and sampling SN
  model parameters given photometric light curve data.

- **Dust laws:** Fast implementations of several commonly used
  extinction laws; can be used to construct SN models that include dust.

- **I/O:** Convenience functions for reading and writing peculiar data
  formats used in other packages and getting dust values from
  SFD (1998) maps.

- **Built-in supernova models** such as SALT2, MLCS2k2, Hsiao, Nugent,
  PSNID, SNANA and Whalen models, as well as a variety of built-in
  bandpasses and magnitude systems.

- **Extensible:** New models, bandpasses, and magnitude systems can be
  defined, using an object-oriented interface.


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
  with a (mostly) different feature set. SNCosmo is based on spectral
  timeseries models whereas SNooPy is more focussed on models of light
  curves in given bands.


.. _`snfit`: http://supernovae.in2p3.fr/salt
.. _`SNANA`: http://sdssdp62.fnal.gov/sdsssn/SNANA-PUBLIC/
.. _`SNooPy`: http://csp.obs.carnegiescience.edu/data/snpy


The name SNCosmo
================

A natural choice, "snpy", was already taken (`SNooPy`_) so I tried to
be a little more descriptive. The package is really specific to
supernova *cosmology*, as it doesn't cover other types of supernova
science (radiative transfer simulations for instance).  Hence
"sncosmo".


Contributors
============

Alphabetical by last name:

* Kyle Barbary
* Tom Barclay
* Rahul Biswas
* Matt Craig
* Ulrich Feindt
* Brian Friesen
* Danny Goldstein
* Saurabh Jha
* Steve Rodney
* Caroline Sofiatti
* Rollin C. Thomas
* Michael Wood-Vasey
