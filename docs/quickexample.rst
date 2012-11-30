A quick example
---------------

  >>> import numpy as np
  >>> import snsim
  >>> 
  >>> # Describe the survey
  >>> survey_observations = {'date': [53100., 53100., ... ], 
  >>>                        'band': ['DECam::DESg', 'DECam::DESr', ... ], 
  >>>                        ... }
  >>> survey = snsim.Survey(survey_observations)
  >>> 
  >>> # Describe the things we're looking for.
  >>> sn1a_model = snsim.SALT2Model()
  >>> 
  >>> # Describe the distribution of the parameters of the model.
  >>> def param_gen():
  >>>   return {'x1': np.random.uniform(),
  >>>           'c': np.random.uniform()}
  >>> 
  >>> # Describe how often the transients occur.
  >>> def vrate(z):
  >>>     return 0.25e-4 * (1. + 2. * z)
  >>> 
  >>> # Run the simulation
  >>> survey.simulate(sn1a_model, vrate=vrate, param_gen=param_gen)

