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

Print a summary of the model
----------------------------

    >>> print model
    Model class: TimeSeriesModel
    Model name: hsiao
    Model version: 3.0
    Restframe phases: [-20, .., 85] days (106 points)
    Restframe dispersion: [1000, .., 25000] Angstroms (2401 points) 
    Reference phase: 0.0 days
    Cosmology: WMAP9(H0=69.3, Om0=0.286, Ode0=0.713)
    Parameters:
        fscale=1.0 [ absmag=None ]
        z=None
	c=None

Retrieve native model phases and wavelengths
--------------------------------------------

    >>> model.times() # native model phase values (in days)
    array([-20., -19., -18., ..., 83.,  84.,  85.])
    >>> model.disp() # native model wavelength values (in angstroms)
    array([  1000.,   1010.,   1020., ...,  24980.,  24990.,  25000.])


Retrieving Model Spectral Values
--------------------------------
To retrieve the spectral values of the model (in ergs / s / cm^2 / Angstrom)
at a given phase: 

    >>> model.flux(-10.5)  # returns an array of shape (2401,)
    array([  1.08078008e-12,   1.31328504e-12,   1.60128855e-12, ...,
             1.45621948e-11,   1.45639898e-11,   1.45656013e-11])

These flux values correspond to the native model dispersion values. To specify
different dispersion values (in Angstroms):

    >>> model.flux(-10.5, [3000., 4000.])
    array([  2.56244970e-09,   3.07976167e-09])

You can also specify multiple phase values:

    >>> model.flux([-10.5, -9.4], [3000., 4000.])
    array([[  2.56244970e-09,   3.07976167e-09],
           [  3.22637597e-09,   3.83064860e-09]])

or set the phase to `None` to get all the native phase values:

    >>> model.flux(None, 4000.)  # returns an array with shape (106, 1)
    array([[  1.30002859e-38],
           [  9.90476473e-13],
           [  9.13543963e-11],
	   ...

Note that the latter is in fact a monochromatic light curve at a wavelength of
4000 Angstroms.

This will retrieve *all* the model flux values:

    >>> model.flux()  # Returns an array of shape (106, 2401)


Normalizing the Model
---------------------

The spectral flux values returned above are the "true" model
values. No scaling has been assumed or applied. Not all models will be
scaled consistently, so it is convenient to be able to scale the flux
of the model. This can be done by setting either the 'fscale' or
'absmag' parameter of the model. For example,

    >>> model.flux(0., 4000.)  # before scaling, 'fscale' is 1. 
    8.2949433988233068e-09
    >>> model.set(fscale=2.)  # simply sets the parameter.
    >>> model.flux(0., 4000.)  # now the values are twice as big.
    1.6589886797646614e-08

It is often more convenient to be able to normalize the model so that the peak
magnitude in some band is some desired value. The following sets ``fscale``
so that the absolute AB magnitude is -19.3 in the Bessell B band at the
"reference phase" of the model (in this case, the reference phase is 0 days).

    >>> model.set(absmag=-19.3)
    >>> model.params['fscale']  # flux scale has been changed.
    46846229.523142166

You can change the reference phase of the model, and it will be used in 
subsequent calls setting the absolute magnitude:

    >>> model.refphase = -1.
    >>> model.set(absmag=(-19.3, 'bessellb', 'ab'))
    >>> model.params['fscale']
    47183480.795604385


Redshifting the model
---------------------

The redshift of the model can be set, and the model phases, wavelengths,
and flux values will be adjusted appropriately.

    >>> model.set(z=1.)
    >>> model.times()  # observer-frame phases.
    array([ -40.,  -38.,  -36., ..., 166.,  168.,  170.])
    >>> model.disp()  # observer frame dispersion.
    array([  2000.,   2020.,   2040., ...,  49960.,  49980.,  50000.])

You can still get the rest-frame values for the phases and dispersion by using
the restframe keyword:

    >>> model.times(modelframe=True)
    array([ -20.,  -19.,  -18., ..., 83.,  84.,  85.])

The flux density has been scaled as if the source was at a luminosity distance
corresponding to ``z=1`` (and note that the wavelength input is 4000 Angstroms
in the observer frame (2000 Angstroms rest-frame):

    >>> model.flux(0., 4000.)
    2.4289400099571839e-20

The luminosity distance is calculated using the `astropy.cosmology` package.
By default, the cosmology is set to a flat WMAP9 cosmology.

    >>> model.cosmo
    WMAP9(H0=69.3, Om0=0.286, Ode0=0.713)

You can also set the current cosmology:

    >>> from astropy.cosmology import FlatLambdaCDM
    >>> model.cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
    >>> model.cosmo
    FlatLambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

You can also set the cosmology to `None`, which will make it so that the 
luminosity distance is not applied.

    >>> model.cosmo = None

The model will still be redshifted, but the flux will be scaled so
that the bolometric flux (ergs / s / cm^2) remains constant.


Setting other model parameters
------------------------------

Some types of models have more parameters. For example, take the SALT2
model, where parameters ``x1`` and ``c`` determine the shape of the
spectral time series:

    >>> model = sncosmo.get_model('salt2')
    >>> print model
    Model class: SALT2Model
    Model name: salt2
    Model version: 2.0
    Restframe phases: [-20, .., 50] days (71 points)
    Restframe dipsersion: [2000, .., 9200] Angstroms (721 points) 
    Reference phase: 0.0 days
    Cosmology: WMAP9(H0=69.3, Om0=0.286, Ode0=0.713)
    Current Parameters:
        fscale=1.0 [ absmag=None ]
        z=None
        c=0.0
        x1=0.0

You can see that in addition to ``fscale``, ``absmag``, and ``z``, there
are also ``c`` and ``x1`` (both set to 0. by default). To set these parameters
to other values, use the ``set`` method:

    >>> model.set(c=0.1, x1=-0.5)

To get the current value of any of the parameters:

    >>> model.params
    OrderedDict([('fscale', 1.0), ('mabs', None), ('z', None),
                 ('c', 0.1), ('x1', -0.5)])
    >>> model.params['c']
    0.1

Note that the ``params`` attribute *accesses* the parameter values,
but cannot be used to set them. This will not complain:

    >>> model.params['c'] = 0.2  # wrong; doesn't change model params

but the model's parameters will remain unchanged:

    >>> model.params['c']
    >>> 0.1

Instead, to *set* the parameters, the ``set`` method must be used.

Generating synthetic photometry
-------------------------------

There are two methods that generate synthetic photometry from the model
spectra: ``flux`` and ``mag``. Continuing on with our SALT2 model, let's
set the absolute magnitude and redshift:

    >>> model.set(absmag=(-19.3, 'bessellb', 'ab'), z=0.5)

Flux
````

To get the flux (photons / s / cm^2) in the SDSS i band at a phase of 0 days:

    >>> model.bandflux(0., 'sdssi')
    0.00032041370572056057

This method also accepts numpy arrays or lists:

    >>> model.bandflux([0., 0., 1., 1.], ['sdssi', 'sdssz', 'sdssi', 'sdssz'])
    array([  3.20413706e-04,   5.72410077e-05,   3.20367693e-04,
             5.74384657e-05])

Broadcasting is also supported:

    >>> model.bandflux([0., 1.], 'sdssi')
    array([ 0.00032041,  0.00032037])

We have been specifying the bandpasses as strings (``'sdssi'`` and
``'sdssz'``).  This works because these bandpasses are built-in: these
are strings that correspond to built-in `~sncosmo.Bandpass` objects
(see the :ref:`list-of-built-in-bandpasses`). The flux method also
directly accepts actual `~sncosmo.Bandpass` objects. First, we
construct a custom bandpass:

    >>> from sncosmo import Bandpass
    >>> disp = np.array([4000., 4200., 4400., 4600., 4800., 5000.])
    >>> trans = np.array([0., 1., 1., 1., 1., 0.])
    >>> band = Bandpass(disp, trans, name='tophatg')
    >>> model.bandflux([0., 1.], band)
    array([ 0.00013845,  0.0001293 ])

Magnitude
`````````

Magnitude works very similarly to flux. The only difference is that in
addition to a bandpass, a magnitude system must also be specified.

    >>> model.bandmag([0., 1.], 'sdssi', 'ab')
    array([ 22.6255077 ,  22.62566363])
    >>> model.bandmag([0., 1.], 'sdssi', 'vega')
    array([ 22.26843273,  22.26858865])

The magnitude systems work similarly to bandpasses: ``'ab'`` and
``'vega'`` refer to built-in `~sncosmo.MagSystem` objects, but you can
also directly supply custom `~sncosmo.MagSystem` objects.

Creating models directly (not from built-in data)
-------------------------------------------------


Creating New Models Classes
---------------------------

In this package, a "model" is something that specifies the spectral
timeseries as a function of an arbitrary number of parameters. For
example, the SALT2 model has two parameters (`x1` and `c`) that
determine a unique spectrum as a function of phase. New models can be
easily implemented by deriving from the abstract base class
`sncosmo.Model` and inheriting most of the functionality described here.


.. _list-of-built-in-models:

List of Built-in Models
-----------------------

.. automodule:: sncosmo._builtin.models
