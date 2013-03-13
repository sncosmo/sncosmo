Reference/API
=============

Models
------

.. autosummary::
   :toctree: _generated

   sncosmo.get_model
   sncosmo.Model
   sncosmo.TimeSeriesModel
   sncosmo.StretchModel
   sncosmo.SALT2Model


.. inheritance-diagram:: sncosmo.StretchModel sncosmo.SALT2Model 

Spectral tools
--------------

.. autosummary::
   :toctree: _generated

   sncosmo.get_bandpass
   sncosmo.get_magsystem
   sncosmo.Bandpass
   sncosmo.Spectrum
   sncosmo.MagSystem
   sncosmo.SpectralMagSystem
   sncosmo.ABMagSystem
   
.. inheritance-diagram:: sncosmo.ABMagSystem sncosmo.SpectralMagSystem

Registry (:mod:`sncosmo.registry`)
----------------------------------

.. currentmodule:: sncosmo

.. autosummary::
   :toctree: _generated

   registry.register_loader
   registry.register
   registry.retrieve

Utilities (:mod:`sncosmo.utils`)
--------------------------------

.. autosummary::
   :toctree: _generated

   utils.read_griddata
   utils.GridData
   utils.extinction_ratio_ccm
