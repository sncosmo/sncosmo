
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

Functions
=========

.. autosummary::
   :toctree: _generated

   get_model
   get_bandpass
   get_magsystem
   extinction_ccm
   registry.register_loader
   registry.register
   registry.retrieve
   fit_model
   readlc
   writelc
   plotlc

Classes
=======

.. autosummary::
   :toctree: _generated

   TimeSeriesModel
   StretchModel
   SALT2Model
   Bandpass
   SpectralMagSystem
   ABMagSystem
   PhotoTyper
