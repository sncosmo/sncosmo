Reference/API
=============

Models
------

.. autosummary::
   :toctree: _generated

   sncosmo.get_model
   sncosmo.Model
   sncosmo.TimeSeriesModel
   sncosmo.SALT2Model


.. inheritance-diagram:: sncosmo.TimeSeriesModel sncosmo.SALT2Model

Spectral tools
--------------

.. autosummary::
   :toctree: _generated

   sncosmo.Bandpass
   sncosmo.Spectrum
   sncosmo.MagnitudeSystem
   sncosmo.ABSystem

.. inheritance-diagram:: sncosmo.ABSystem

Registry (:mod:`sncosmo.registry`)
----------------------------------

.. currentmodule:: sncosmo

.. autosummary::
   :toctree: _generated

   registry.register_loader
   registry.retrieve

Utilities (:mod:`sncosmo.utils`)
--------------------------------

.. autosummary::
   :toctree: _generated

   utils.read_griddata
   utils.GridData
