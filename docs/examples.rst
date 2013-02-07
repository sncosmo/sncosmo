********
Examples
********

.. note:: Disregard these examples for now. They are out of date with the
          rapidly changing API.

Use supernova models
====================

Redshift a spectrum and perform synthetic photometry
====================================================


Simulate supernova observations
===============================

We want to simulate observations of supernovae detected by a very
short survey. The survey has 3 nights of observations in two
filters each night. First, read in information about the observations
from a file using the :mod:`astropy.table` module::

  from astropy.table import Table
  from StringIO import StringIO
  obsfile = StringIO(
    'field    date        band ccdgain ccdnoise skysig psfsig   zp zpsys\n'
    '    1 56293.0 DECam::DESg     1.0     10.0   50.0    2.0 25.0    ab\n'
    '    1 56293.1 DECam::DESr     1.0     10.0   50.0    2.0 25.0    ab\n'
    '    1 56398.0 DECam::DESg     1.0     10.0   50.0    2.0 25.0    ab\n'
    '    1 56298.1 DECam::DESr     1.0     10.0   50.0    2.0 25.0    ab\n'
    '    1 56203.0 DECam::DESg     1.0     10.0   50.0    2.0 25.0    ab\n'
    '    1 56303.1 DECam::DESr     1.0     10.0   50.0    2.0 25.0    ab\n')
  obs = Table.read(obsfile, format='ascii')

Next, we need to define things about the fields we observed. In this case we
only observed a single field, with ID ``1``. We can define various properties
of the observed fields, but minimally we have to define the area
(in square degrees)::

  fields = {1: {'area': 2.75}}

Our observations were made through certain bandpasses, which we have
labelled as ``'DECam::DESg'``, for example. We have to define the
bandpass implied by each label, using a :mod:`sncosmo.Bandpass` object
for each one. If the bandpasses are built-in ones, we can get them
using the factory function ``sncosmo.bandpass``::

  import sncosmo
  bandpasses = {}
  for name in ['DECam::DESg', 'DECam::DESr', 'Bessell::B']:
      bandpasses[name] = sncosmo.bandpass(name)

Note that we have also read in the Bessell B bandpass. We'll need that later.

Similar to the filters we've used, the zeropoint of each observation
is also defined by a simple label, such as ``'ab'``. We need to read
in the spectrum that defines each zeropoint system. The built-in
spectra are ``'ab'`` and ``'vega'``. ::

  zpspectra = {'ab': sncosmo.spectrum('ab')}

We can now initialize our survey object::

  short_survey = sncosmo.Survey(fields, obs, bandpasses, zpspectra)

This object contains information on all our observations and observed
fields, and it knows about the :mod:`sncosmo.Bandpass` and
:mod:`sncosmo.Spectrum` objects that define our observed filters and
zeropoint systems.

Before simulating what is detected by the survey, we need to define what kind of transients we are trying to find! This is done using a ``sncosmo.Model`` object. To get a simple built-in Type Ia supernova model::

  snmodel = sncosmo.model('hsiao')

We can now run the simulation::

  sne = short_survey.simulate(snmodel, params={'m':-19.3},
                              mband='Bessell::B', zpsys='ab',
                              vrate=0.25e-4, z_range=(0., 2.0), z_bins=20,
			      verbose=True)
  print "found", len(sne), "sne"

The supernova models are arbitrarily normalized, by design. This
allows the user the flexibility to specify the normalization as
desired. The three arguments ``params``, ``mband`` and ``zpsys``
specify the normalization explicitly: Here, the peak magnitude should
be ``-19.3`` in the ``Bessell::B`` rest-frame band and the ``'ab'``
magnitude system.

The volumetric supernova rate is given by ``vrate``. Here it is
``0.25 * 10**-4 SNe yr^-1 Mpc^-3``.




Vary the SN rate with redshift
------------------------------

Rather than using a constant volumetric rate, the SN rate can be a function of redshift. To do this, define a function that returns the rate given the redshift::

  def sn1a_rate(z):
      if z < 1:
          return 0.25e-4 * (1 + 2.5 * z)
      else:
          return 0.25e-4 * 3.5

Pass the function to the ``vrate`` parameter::

  sne = short_survey.simulate(snmodel, params={'m':-19.3},
                              mband='Bessell::B', zpsys='ab',
                              vrate=sn1a_rate, z_range=(0., 2.0), z_bins=20,
			      verbose=True)

The supernova models are arbitrarily normalized, by design. This allows the user the flexibility to specify the normalization as desired. The three arguments ``params``, ``mband`` and ``zpsys`` specify the normalization explicitly: Here, the peak magnitude should be ``-19.3`` in the ``Bessell::B`` rest-frame band and the ``'ab'`` magnitude system.


Use a distribution of SN model parameters
-----------------------------------------

Define a function that returns a dictionary of randomly select
parameters on each call::

  def param_gen():
      return {'m': np.random.normal(-19.3, 0.15)}

Pass this to the ``params`` parameter::

  sne = short_survey.simulate(snmodel, params=param_gen,
                              mband='Bessell::B', zpsys='ab',
                              vrate=sn1a_rate, z_range=(0., 2.0), z_bins=20,
			      verbose=True)
