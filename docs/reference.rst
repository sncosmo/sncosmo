
***************
Reference / API
***************

Built-ins
=========

.. toctree::
   :maxdepth: 1

   builtins/models
   builtins/bandpasses
   builtins/magsystems

.. currentmodule:: sncosmo

Models
======

.. autosummary::
   :toctree: _generated

   get_sourcemodel
   SourceModel
   TimeSeriesModel
   StretchModel
   SALT2Model
   ObsModel

Extinction Functions & PropagationEffects
=========================================

*Functions related to interstellar dust extinction*

.. autosummary::
   :toctree: _generated

   extinction
   get_ebv_from_map
   PropagationEffect
   RvDust
   CCM89Dust
   OD94Dust
   F99Dust


Bandpasses
==========

.. autosummary::
   :toctree: _generated

   get_bandpass
   Bandpass

Magnitude Systems
=================

.. autosummary::
   :toctree: _generated

   get_magsystem
   MagSystem
   ABMagSystem
   SpectralMagSystem

Data I/O
========

*Convenience functions for reading and writing photometric data to
standard (and not-so-standard) formats*

.. autosummary::
   :toctree: _generated

   read_lc
   write_lc
   read_bandpass
   load_example_data
   read_snana_ascii
   read_snana_fits
   read_snana_simlib
   

Fitting & Typing Photometric Data
=================================

*Fit model parameters to photometric data & compare models to data*

.. autosummary::
   :toctree: _generated

   fit_lc
   nest_lc
   mcmc_lc
   PhotoTyper

Plotting
========

*Convenience functions for quick standard plots (requires matplotlib)*

.. autosummary::
   :toctree: _generated

   plot_lc
   animate_model

Registry
========

*Customizing "built-in" models, bandpasses, and magnitude systems*

.. autosummary::
   :toctree: _generated

   registry.register_loader
   registry.register
   registry.retrieve

Class Inheritance Diagrams
==========================

.. inheritance-diagram:: SourceModel TimeSeriesModel StretchModel SALT2Model
   :parts: 1

.. inheritance-diagram:: PropagationEffect RvDust F99Dust OD94Dust CCM89Dust
   :parts: 1

.. inheritance-diagram:: MagSystem ABMagSystem SpectralMagSystem
   :parts: 1

.. inheritance-diagram:: Bandpass
   :parts: 1
