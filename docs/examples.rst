Examples
========

A quick example
---------------

  >>> import numpy as np
  >>> import snsim
  >>>  
  >>> # Describe the survey
  >>> survey_observations = {'date': [53100., 53100., ... ], 
  >>>                        'band': ['DECam::DESg', 'DECam::DESr', ... ], 
  >>>                        ... }
  >>> survey = snsim.Survey(fields=survey_fields,
  >>>                       obs=survey_observations,
  >>>                       bandpasses=survey_bandpasses
  >>>                       zpspectra=zpspectra)
  >>> 
  >>> 
  >>> snmodel = snsim.model('salt2')
  >>> params = {'m': -19.3}
  >>> 
  >>> 
  >>> survey.simulate(snmodel, params, mband='Bessell::B', vrate=1.e-4)


Vary the SN rate with redshift
------------------------------

Define a function that returns the rate as a function of redshift:

  >>> def sn1a_rate(z):
  >>>     if z < 1:
  >>>         return 0.25e-4 * (1 + 2.5 * z)
  >>>     else:
  >>>         return 0.25e-4 * 3.5
  >>> 
  >>> survey.simulate(sn1a_model, params, vrate=sn1a_vrate)


Using a distribution of SN model parameters
-------------------------------------------

Define a function that returns the parameters:

  >>> def param_gen():
  >>>   return {'m': -19.3,
  >>>           'x1': np.random.uniform(),
  >>>           'c': np.random.uniform()}

