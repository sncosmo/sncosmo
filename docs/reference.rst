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

Plotting
--------

.. autosummary::
   :toctree: _generated

   sncosmo.plotlc

Reading / Writing data
----------------------

.. autosummary::
   :toctree: _generated

   sncosmo.readlc
   sncosmo.writelc

Fitting data
------------

.. autosummary::
   :toctree: _generated

   sncosmo.fit_model

Photometric Typing (:mod:`sncosmo.typing`)
------------------------------------------
.. currentmodule:: sncosmo

.. autosummary::
   :toctree: _generated

   typing.evidence
   typing.PhotoTyper


Registry (:mod:`sncosmo.registry`)
----------------------------------

.. currentmodule:: sncosmo

.. autosummary::
   :toctree: _generated

   registry.register_loader
   registry.register
   registry.retrieve

Other
-----

.. autosummary::
   :toctree: _generated

   sncosmo.extinction.extinction_ccm

