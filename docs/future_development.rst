*************************************
Future features and development plans
*************************************

Testing
-------

Unit testing should be implemented.

Data Format
-----------

Functions that accept data currently require it to be either a numpy
structured array or dictionary of 1-d numpy arrays (something that can
be accessed with ``data['field']``). The field names must strictly
include ``'time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'``. It would
be nice to allow a wider variety of field names (while still being
unique).

Data Covariance
---------------

There is no way to specify covariance between data points.

Model and spectra data hosting/fallbacks
----------------------------------------

Currently, data for "built-in" models and spectra are hosted in a
variety of places (the original provider's origin, if possible).  (1)
It would be ideal to have a central place to host the data
(data.astropy.org/ perhaps) (2) There should be a local fallback
option.

Model extrapolation in time
---------------------------

Currently, outside the phase range of a model, the flux is effectively
zero. It would be nice to have an option to extend the model with some
exponential falloff.

Model interpolation
-------------------

Currently, models are interpolated using a second order spline. Add an
option for interpolating using other degree splines.

SALT2 dispersion model
----------------------

The SALT2 model for describing dispersion in light curves has not been
implemented.


Errors in fit parameters
------------------------

``fit_model`` returns best-fit parameters, but no uncertainties.

Fitting spectra
---------------

``fit_model`` fits light curve data. A similar function that fits
a spectrum would be useful.
