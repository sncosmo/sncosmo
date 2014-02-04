
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

Model, Sources & Effects
========================

.. autosummary::
   :toctree: api

   Model
   Source
   TimeSeriesSource
   StretchSource
   SALT2Source
   CCM89Dust
   OD94Dust
   F99Dust
   get_source

Extinction
==========

*Functions related to interstellar dust extinction*

.. autosummary::
   :toctree: api

   extinction
   get_ebv_from_map

Bandpasses
==========

.. autosummary::
   :toctree: api

   get_bandpass
   Bandpass

Magnitude Systems
=================

.. autosummary::
   :toctree: api

   get_magsystem
   MagSystem
   ABMagSystem
   SpectralMagSystem

Data I/O
========

*Convenience functions for reading and writing photometric data to
standard (and not-so-standard) formats*

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
   write_griddata_fits

Fitting Photometric Data
========================

*Fit model parameters to photometric data*

.. autosummary::
   :toctree: api

   fit_lc
   nest_lc
   mcmc_lc

Plotting
========

*Convenience functions for quick standard plots (requires matplotlib)*

.. autosummary::
   :toctree: api

   plot_lc
   animate_model

Simulation
==========

.. autosummary::
   :toctree: api

   simulate_vol

Registry
========

*Connecting strings to models, bandpasses, and magnitude systems*

.. autosummary::
   :toctree: api

   registry.register_loader
   registry.register
   registry.retrieve

Class Inheritance Diagrams
==========================

.. inheritance-diagram:: Source TimeSeriesSource StretchSource SALT2Source
   :parts: 1

.. inheritance-diagram:: PropagationEffect RvDust F99Dust OD94Dust CCM89Dust
   :parts: 1

.. inheritance-diagram:: MagSystem ABMagSystem SpectralMagSystem
   :parts: 1

