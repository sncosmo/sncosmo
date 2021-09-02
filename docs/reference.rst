
***************
Reference / API
***************

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
   MLCS2k2Source
   SALT2Source
   SALT3Source
   SNEMOSource
   SUGARSource
   
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
   AggregateBandpass
   BandpassInterpolator
   MagSystem
   ABMagSystem
   SpectralMagSystem
   CompositeMagSystem

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
   load_example_spectrum_data
   read_snana_ascii
   read_snana_fits
   read_snana_simlib
   read_griddata_ascii
   read_griddata_fits
   write_griddata_ascii
   write_griddata_fits

Spectra
=======

.. autosummary::
   :toctree: api

   Spectrum


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

   select_data
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

.. inheritance-diagram:: models magsystems bandpasses
   :parts: 1
