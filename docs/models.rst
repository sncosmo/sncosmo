****************
Supernova Models
****************

Creating a model
----------------

There are several built-in models available by name. These models can be
retrieved using the `sncosmo.get_model` function:

    >>> import sncosmo
    >>> model = sncosmo.get_model('hsiao')
    >>> model
    <TimeSeriesModel 'hsiao' version='3.0' at 0x3daf8d0>
    >>> print model
    Model class: TimeSeriesModel
    Model name: hsiao
    Model version: 3.0
    Restframe phases: [-20, .., 85] days (106 points)
    Restframe dispersion: [1000, .., 25000] Angstroms (2401 points) 
    Reference phase: 0.0 days
    Cosmology: WMAP9(H0=69.3, Om0=0.286, Ode0=0.713)
    Parameters:
        flux_scale=1.0 [ absmag=None ]
        z=None
	c=None

.. note:: In fact, the model data is hosted remotely, downloaded
          as needed, and cached locally. So the first time you
          load a given model using `~sncosmo.get_model`, you need to be
          connected to the internet.  You will see a progress bar as
          the data are downloaded.

There are multiple versions of some models. To get a different
version, you can do:

    >>> model = sncosmo.get_model('hsiao', version='2.0')

If ``version`` is excluded, the latest version is used. 
See the :ref:`list-of-built-in-models` below.

Retrieving Model Spectral Values
--------------------------------
To retrieve the spectral values of the model (in ergs / s / cm^2 / Angstrom)
at a given phase: 

    >>> model.flux_density(-10.5)  # returns an array of shape (2401,)
    array([  1.08078008e-12,   1.31328504e-12,   1.60128855e-12, ...,
             1.45621948e-11,   1.45639898e-11,   1.45656013e-11])

These flux values correspond to the native model dispersion values. To specify
different dispersion values (in Angstroms):

    >>> model.flux_density(-10.5, [3000., 4000.])
    array([  2.56244970e-09,   3.07976167e-09])

You can also specify multiple phase values:

    >>> model.flux_density([-10.5, -9.4], [3000., 4000.])
    array([[  2.56244970e-09,   3.07976167e-09],
           [  3.22637597e-09,   3.83064860e-09]])

or set the phase to `None` to get all the native phase values:

    >>> model.flux_density(None, 4000.)  # returns an array with shape (106, 1)
    array([[  1.30002859e-38],
           [  9.90476473e-13],
           [  9.13543963e-11],
	   ...

Note that the latter is in fact a monochromatic light curve at a wavelength of
4000 Angstroms.

This will retrieve *all* the model flux values:

    >>> model.flux_density()  # Returns an array of shape (106, 2401)


Normalizing the Model
---------------------

The spectral flux values returned above are the "true" model
values. No scaling has been assumed or applied. Not all models will be
scaled consistently, so it is convenient to be able to scale the flux
of the model. This can be done by setting either the 'flux_scale' or
'absmag' parameter of the model. For example,

    >>> model.flux_density(0., 4000.)  # before scaling, 'flux_scale' is 1. 
    8.2949433988233068e-09
    >>> model.set(flux_scale=2.)  # simply sets the parameter.
    >>> model.flux_density(0., 4000.)  # now the values are twice as big.
    1.6589886797646614e-08

It is often more convenient to be able to normalize the model so that the peak
magnitude in some band is some desired value. The following sets ``flux_scale``
so that the absolute AB magnitude is -19.3 in the Bessell B band at the
"reference phase" of the model (in this case, the reference phase is 0 days).

    >>> model.set(absmag=(-19.3, 'bessellb', 'ab'))
    >>> model.params['flux_scale']  # flux scale has been changed.
    46846229.523142166

You can change the reference phase of the model, and it will be used in 
subsequent calls setting the absolute magnitude:

    >>> model.refphase = -1.
    >>> model.set(absmag=(-19.3, 'bessellb', 'ab'))
    >>> model.params['flux_scale']
    47183480.795604385

Retrieving model phases and wavelengths
---------------------------------------

    >>> model.phases() # native model phase values (in days)
    array([-20., -19., -18., ..., 83.,  84.,  85.])
    >>> model.dispersion() # native model wavelength values (in angstroms)
    array([  1000.,   1010.,   1020., ...,  24980.,  24990.,  25000.])

Redshifting the model
---------------------

The redshift of the model can be set, and the model phases, wavelengths,
and flux values will be adjusted appropriately.

    >>> model.set(z=1.)
    >>> model.phases()  # observer-frame phases.
    array([ -40.,  -38.,  -36., ..., 166.,  168.,  170.])
    >>> model.dispersion()  # observer frame dispersion.
    array([  2000.,   2020.,   2040., ...,  49960.,  49980.,  50000.])


The flux density has been scaled as if the source was at a luminosity distance
corresponding to ``z=1`` (and note that the wavelength input is 4000 Angstroms
in the observer frame (2000 Angstroms rest-frame):

    >>> model.flux_density(0., 4000.)
    2.4289400099571839e-20

The luminosity distance is calculated using a `~astropy.cosmology.Cosmology`
instance. You can see the current cosmology:

    >>> model.cosmo
    WMAP9(H0=69.3, Om0=0.286, Ode0=0.713)

You can also set the current cosmology:

    >>> from astropy.cosmology import FlatLambdaCDM
    >>> model.cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

You can also set the cosmology to `None`, which will make it so that the 
luminosity distance is not applied. The model will still be redshifted, but
the flux will be scaled so that the bolometric flux (ergs / s / cm^2) remains
constant:

    >>> model.cosmo = None
   

Creating New Models
-------------------

A model is anything where the spectrum as a function of phase can be
parameterized by an arbitrary number of parameters. For example, the
Hsiao and Nugent SN templates are zero parameter models (not counting
amplitude): there is a single spectrum for a given phase. The SALT2
model has two parameters (`x1` and `c`) that determine a unique
spectrum as a function of phase. The Hsiao and Nugent models can be
described using a single class whereas the SALT2 model needs a
separate subclass. All models derive from a common abstract base
class.

.. _list-of-built-in-models:

List of Built-in Models
-----------------------

.. automodule:: sncosmo._builtin.models
