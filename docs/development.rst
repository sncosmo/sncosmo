***********
Development
***********

Bug reports, comments, and help with development are very welcome.
Source code and issue tracking is hosted on github:
https://github.com/kbarbary/sncosmo

Stability
=========

The basic features of the models, bandpasses and magnitude systems
(documented below) can be considered "fairly" stable (but no
guarantees).  The fitting and typing functionalities are more
experimental and the API may change as it gets more real-world
testing.

Future Plans
============

Testing
-------

Unit testing should be implemented.

Data Covariance
---------------

There is currently no way to specify covariance between data points.

Model and spectra data hosting/fallbacks
----------------------------------------

Currently, data for "built-in" models and spectra are hosted in a
variety of places (the original provider's origin, if possible).

* It would be ideal to have a central place to host the data
(data.astropy.org/ perhaps)
* There should be a local fallback option.

Model extrapolation in time
---------------------------

Currently, outside the phase range of a model, the flux is effectively
constant (at the value of the nearest model phase). It would be nice
to have an option to extend the model with some exponential falloff.

Model interpolation
-------------------

Currently, models are interpolated using a second order spline. Add an
option for interpolating using other degree splines.

SALT2 dispersion model
----------------------

The SALT2 model for describing dispersion in light curves has not been
implemented.

Fitting spectra
---------------

``fit_model`` fits light curve data. A similar function that fits
a spectrum would be useful.
