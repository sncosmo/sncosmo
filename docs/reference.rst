
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


Fitting & Typing Photometric Data
=================================

.. autosummary::
   :toctree: _generated

   fit_model
   PhotoTyper

Plotting
========

.. autosummary::
   :toctree: _generated

   plotlc
   animate_model

Registry
========

.. autosummary::
   :toctree: _generated

   registry.register_loader
   registry.register
   registry.retrieve
