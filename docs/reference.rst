
***************
Reference / API
***************

Built-ins
=========

.. toctree::
   :maxdepth: 1

   builtins/sources
   builtins/bandpasses
   builtins/magsystems

.. currentmodule:: sncosmo

Model & Components
==================

.. autosummary::
   :toctree: api

   Model

*Source component of Model*

.. autosummary::
   :toctree: api

   Source
   TimeSeriesSource
   StretchSource
   SALT2Source

*Effect components of Model: interstellar dust extinction*

.. autosummary::
   :toctree: api

   PropagationEffect
   CCM89Dust
   OD94Dust
   F99Dust

Bandpass & Magnitude Systems
============================

.. autosummary::
   :toctree: api

   Bandpass
   MagSystem
   ABMagSystem
   SpectralMagSystem

I/O
===

*Functions for reading and writing photometric data, gridded data, extinction
maps, and more.*

.. autosummary::
   :toctree: api

   read_lc
   write_lc
   read_bandpass
   load_example_data
   read_snana_ascii
   read_snana_fits
   read_snana_simlib
   read_griddata_ascii
   read_griddata_fits
   write_griddata_ascii
   write_griddata_fits

.. _fitting-api:

Fitting Photometric Data
========================

*Estimate model parameters from photometric data*

.. autosummary::
   :toctree: api

   fit_lc
   mcmc_lc
   nest_lc

*Convenience functions*

.. autosummary::
   :toctree: api

   chisq
   flatten_result


Plotting
========

*Convenience functions for quick standard plots (requires matplotlib)*

.. autosummary::
   :toctree: api

   plot_lc


Simulation
==========

.. autosummary::
   :toctree: api

   zdist
   realize_lcs

Registry
========

*Register and retrieve custom built-in sources, bandpasses, and
magnitude systems*

.. autosummary::
   :toctree: api

   register
   register_loader
   get_source
   get_bandpass
   get_magsystem

Class Inheritance Diagrams
==========================

.. inheritance-diagram:: Source TimeSeriesSource StretchSource SALT2Source
   :parts: 1

.. inheritance-diagram:: PropagationEffect F99Dust OD94Dust CCM89Dust
   :parts: 1

.. inheritance-diagram:: MagSystem ABMagSystem SpectralMagSystem
   :parts: 1
