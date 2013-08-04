
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

Models, Bandpasses & Magnitude Systems
======================================

.. autosummary::
   :toctree: _generated

   get_model
   get_bandpass
   get_magsystem
   TimeSeriesModel
   StretchModel
   SALT2Model
   Bandpass
   SpectralMagSystem
   ABMagSystem
   extinction_ccm

Data I/O
========

.. autosummary::
   :toctree: _generated

   readlc
   writelc
   load_example_data

Fitting & Typing Photometric Data
=================================

.. autosummary::
   :toctree: _generated

   fit_model
   PhotoTyper

Plotting
========

*Simple convenience functions for standard plots using matplotlib*

.. autosummary::
   :toctree: _generated

   plotlc
   plotpdf
   animate_model

Registry
========

.. autosummary::
   :toctree: _generated

   registry.register_loader
   registry.register
   registry.retrieve
