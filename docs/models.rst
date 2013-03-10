****************
Supernova Models
****************

Getting Started
---------------

You can create a model directly from your own data, but there are
several built-in models available by name. These models can be
retrieved using the `sncosmo.get_model` function:

    >>> import sncosmo
    >>> model = sncosmo.get_model('hsiao')
    >>> print model
    Model class: TimeSeriesModel
    Model name: hsiao
    Model version: 3.0
    Restframe phases: [-20, .., 85] days (106 points)
    Restframe dispersion: [1000, .., 25000] Angstroms (2401 points) 
    Reference phase: 0.0 days
    Cosmology: WMAP9(H0=69.3, Om0=0.286, Ode0=0.713)
    Parameters:
        z=None
        fluxscaling=1.0 [ absmag=None ]

.. note:: In fact, the model data is hosted remotely, downloaded
          as needed, and cached locally. So the first time you
          load a given model using `sncosmo.get_model`, you need to be
          connected to the internet.  You will see a progress bar as
          the data are downloaded.

There are multiple versions of some models. To get a different
version, you can do:

    >>> model = sncosmo.get_model('hsiao', version='2.0')

If ``version`` is excluded, the latest version is used. See the bottom of
the page for a list of built-in models.

Retrieving Model Spectra
------------------------

   >>> model.flux_density()


Setting Model Parameters
------------------------


Extending Models
----------------

A model is anything where the spectrum as a function of phase can be
parameterized by an arbitrary number of parameters. For example, the
Hsiao and Nugent SN templates are zero parameter models (not counting
amplitude): there is a single spectrum for a given phase. The SALT2
model has two parameters (`x1` and `c`) that determine a unique
spectrum as a function of phase. The Hsiao and Nugent models can be
described using a single class whereas the SALT2 model needs a
separate subclass. All models derive from a common abstract base
class.


Built-in Models
---------------

.. automodule:: sncosmo._builtin.models
