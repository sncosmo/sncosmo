
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

*Core module classes and functions*

.. autosummary::
   :toctree: _generated

   get_model
   get_bandpass
   get_magsystem
   Model
   TimeSeriesModel
   StretchModel
   SALT2Model
   Bandpass
   MagSystem
   ABMagSystem
   SpectralMagSystem
   extinction_ccm

Data I/O
========

*Convenience functions for reading and writing photometric data to standard formats*

.. autosummary::
   :toctree: _generated

   read_lc
   write_lc
   load_example_data

Fitting & Typing Photometric Data
=================================

*Fit model parameters to photometric data & compare models to data*

.. autosummary::
   :toctree: _generated

   fit_lc
   mcmc_lc
   PhotoTyper

Plotting
========

*Convenience functions for quick standard plots (requires matplotlib)*

.. autosummary::
   :toctree: _generated

   plot_lc
   plot_pdf
   animate_model

Registry
========

*Customizing "built-in" models, bandpasses, and magnitude systems*

.. autosummary::
   :toctree: _generated

   registry.register_loader
   registry.register
   registry.retrieve
