****************
Supernova Models
****************


Initializing
============

There are several built-in models available by name. These models can be
retrieved using the `sncosmo.get_model` function:

    >>> import sncosmo
    >>> model = sncosmo.get_model('hsiao')
    >>> model
    <StretchModel 'hsiao' version='3.0' at 0x3daf8d0>

Some models have multiple versions of the data, which can be retrieved using
the ``version`` keyword. If the keyword is excluded, the latest version is
returned.

    >>> model = sncosmo.get_model('hsiao', version='2.0')

.. note:: In fact, the model data is hosted remotely, downloaded
          as needed, and cached locally. So the first time you
          load a given model using `~sncosmo.get_model`, you need to be
          connected to the internet.  You will see a progress bar as
          the data are downloaded.


Setting parameters
==================

Each model instance has a set of "current" parameters. For example,
the ``hsiao`` model has the following parameters:

    >>> model.parnames
    ['fscale', 'm', 'mabs', 't0', 'z', 'c', 's']

The first three of these, ``fscale``, ``m``, and ``mabs`` are three
different ways of representing the absolute flux scale of the model.
These three, along with ``t0`` (time of phase=0) and ``z`` (redshift)
are common among all models. The last two, ``c`` and ``s`` are
particular to this specific class of model (``StretchModel``).

To set any of the parameters, specify them as keywords to the ``set`` method:

    >>> model.set(mabs=-19.3, z=0.5, c=0.1, s=1.2)
    >>> # or...
    >>> new_params = {'mabs':-19.3, 'z':0.5, 'c':0.1, 's':1.2}
    >>> model.set(**new_params)

To see the current parameters:

    >>> model.params  # note: cannot be used to set parameters
    OrderedDict([('fscale', 8.2521398493289069e-10), ('m', 22.992512497861608),
                 ('mabs', -19.3), ('t0', 0.0), ('z', 0.5), ('c', 0.1),
                 ('s', 1.2)])

Flux scale parameters: ``fscale``, ``m``, and ``mabs``
------------------------------------------------------

Above, notice that ``fscale`` and ``m`` have been set, even though we
didn't specify them. These three parameters are defined as
follows:

* ``fscale``: flux scale relative to native model flux values. All
  model flux values will be multiplied by this value.

* ``m``: apparent magnitude, in a predefined band, magnitude system
  and phase.  If specified, this is used to calculate ``fscale`` (and
  therefore overrides it if both are specified). That is, ``fscale``
  is set so that the synthetic magnitude in the predefined band and
  magnitude system, at the predefined phase is ``m``.

* ``mabs``: absolute magnitude. Sets ``m`` so that ``m = mabs +
  (distance modulus)`` where distance modulus is determined from the
  redshift. If specified, it is used to determine both ``m`` and ``fscale``
  and therefore overrides them.

The predefined band, magnitudes system and phase can be set and accessed as
parameters as such:

    >>> model.refphase
    0.092137760464381685
    >>> model.refband
    <Bandpass 'bessellb' at 0x20c1ed0>
    >>> model.refmagsys
    <sncosmo.spectral.ABMagSystem object at 0x20c1f10>

By default the reference phase is set to the phase of maximum light in
the reference band when the model is loaded. You can change them:

    >>> model.refphase = 3.
    >>> model.refband = 'sdssg'
    >>> model.refmagsys = 'vega'

    >>> model.refphase
    3.
    >>> model.refband
    <Bandpass 'sdssg' at 0x20c1d50>
    >>> model.refmagsys
    <sncosmo.spectral.SpectralMagSystem object at 0x212ac90>

The luminosity distance is calculated using the `astropy.cosmology` package.
By default, the cosmology is set to a flat WMAP9 cosmology.

    >>> model.cosmo
    WMAP9(H0=69.3, Om0=0.286, Ode0=0.713)

You can also change the cosmology:

    >>> from astropy.cosmology import FlatLambdaCDM
    >>> model.cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
    >>> model.cosmo
    FlatLambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

Native model phases and wavelengths
===================================

The model is defined at discrete phases and wavelengths and interpolation is
used to determine flux values at intermediate phase(s) and wavelength(s). To
see what the native values are:

    >>> model.times(modelframe=True)  # rest-frame phases in days
    array([ -24. ,  -22.8,  -21.6, ..., 99.6, 100.8,  102. ])
    >>> model.times()  # observer-frame
    array([ -36. ,  -34.2,  -32.4, ..., 149.4, 151.2,  153. ])
    >>> model.disp(modelframe=True)  # rest-frame wavelengths in angstroms
    array([  1000.,   1010.,   1020., ...,  24980.,  24990.,  25000.])
    >>> model.disp()  # observer-frame
    array([  1500.,   1515.,   1530., ...,  37470.,  37485.,  37500.])

Retrieving a spectrum
=====================
To retrieve a spectrum (in ergs / s / cm^2 / Angstrom) at a given observer-frame time:

    >>> model.flux(-10.5)  # spectrum at time=-10.5 at all the native wavelength values
    array([  2.98208588e-22,   3.81370282e-22,   4.88207315e-22, ...,
             1.52182808e-20,   1.52257192e-20,   1.52324162e-20])
    >>> model.flux(-10.5, [3000., 4000.]) # ... at just two (observer-frame) wavelengths
    >>> model.flux([-10.5, -9.4], [3000., 4000.]) # ... at two times and two wavelengths
    >>> model.flux(None, 4000.)  # ... at all native phases, single wavelength
    >>> model.flux(None, [3000., 4000.])  # flux at all native phases, two wavelengths
    >>> model.flux()  # All native flux values

The shape of the returned array depends on the input. If there are multiple phases returned, it will be a 2-d array:

    >>> model.flux(-10.5, 4000.) # scalar
    >>> model.flux(-10.5)  # 1-d array, shape=(2401,)
    >>> model.flux(-10.5, [3000., 4000.]) # 1-d array, shape=(2,)
    >>> model.flux([-10.5, -9.4], [3000., 4000.]) # 2-d array, shape=(2, 2)
    >>> model.flux(None, 4000.)  # 2-d array, shape=(106, 1)
    >>> model.flux(None, [3000., 4000.]) # 2-d array, shape=(106, 2)
    >>> model.flux()  # 2-d array, shape (106, 2401)

The above are all for observer-frame times and wavelengths. To
interpret the times and wavelengths as being in the rest-frame, use
the modelframe keyword:

    >>> model.flux(-10.5, modelframe=True)
    array([  3.45329754e-22,   4.36235597e-22,   5.51652443e-22, ...,
             1.61948280e-20,   1.61985494e-20,   1.62019061e-20])

Printing a summary
==================

    >>> print model
    Model class: StretchModel
    Model name: hsiao
    Model version: 3.0
    Model phases: [-20, .., 85] days (106 points)
    Model dispersion: [1000, .., 25000] Angstroms (2401 points) 
    Reference phase: 0.0921378 days
    Cosmology: WMAP9(H0=69.3, Om0=0.286, Ode0=0.713)
    Current Parameters:
        fscale = 8.25213984933e-10
        m = 22.9925124979 [bessellb, ab]
	mabs = -19.3 [bessellb, ab]
        t0 = 0.0
        z = 0.5 [dist. mod. = 42.2925, lum. dist. = 2874.1 Mpc]
        c = 0.1
        s = 1.2


Synthetic photometry
====================

To get the flux (photons / s / cm^2) in the SDSS i band at a phase of 0 days:

    >>> model.bandflux('sdssi', 0.)
    0.00032041370572056057
    >>> model.bandflux(['sdssi', 'sdssz', 'sdssi', 'sdssz'], [0., 0., 1., 1.])
    array([  3.20413706e-04,   5.72410077e-05,   3.20367693e-04,
             5.74384657e-05])
    >>> model.bandflux('sdssi', [0., 1.])
    array([ 0.00032041,  0.00032037])
    >>> model.bandflux('sdssi') # all native phases (length 106 array)
    array([ -2.14661119e-23,   2.80447011e-07,   2.51377548e-06, ...,
             1.74574662e-05,   1.71958548e-05, 1.69633095e-05])

Instead of returning flux in photons / s / cm^2, the flux can be normalized
to a desired zeropoint by specifying the ``zp`` and ``zpsys`` keywords,
which can also be scalars, lists, or arrays.

    >>> model.bandflux('sdssi', [0., 1.], zp=25., zpsys='ab')
    array([ 8.38386893,  8.43995715])

Instead of flux, magnitude can be returned. It works very similarly to flux:

    >>> model.bandmag('sdssi', 'ab', [0., 1.])
    array([ 22.6255077 ,  22.62566363])
    >>> model.bandmag('sdssi', 'vega', [0., 1.])
    array([ 22.26843273,  22.26858865])


Bandpasses & magnitude systems
------------------------------

We have been specifying the bandpasses as strings (``'sdssi'`` and
``'sdssz'``).  This works because these bandpasses are in the sncosmo
"registry". However, this is merely a convenience. In place of
strings, we could have specified the actual `~sncosmo.Bandpass`
objects to which the strings correspond. See :doc:`bandpasses`
for more on how to directly create `~sncosmo.Bandpass`
objects.

The magnitude systems work similarly to bandpasses: ``'ab'`` and
``'vega'`` refer to built-in `~sncosmo.MagSystem` objects, but you can
also directly supply custom `~sncosmo.MagSystem` objects. See
:doc:`magsystems` for details.

Model particulars: ``TimeSeriesModel`` & ``StretchModel``
=========================================================

Different classes of models have a very similar API, but a few aspects
differ, by necessity. For example, you can initialize a model directly
from data (in files on disk or in numpy arrays) rather than using the
built-in model data. The initialization for ``TimeSeriesModel`` is
different from the initialization for ``SALT2Model`` (for example)
because the underlying data are very different.

Here we describe particulars of the ``TimeSeriesModel`` and
``StretchModel`` (which only differ by the addition of a stretch
parameter ``s``).

Initializing
------------

These models can be initialized directly from numpy arrays. Below, we build a
very simple model, of a source with a flat spectrum at all times,
rising from phase -50 to 0, then declining from phase 0 to +50.

    >>> phase = np.linspace(-50., 50., 11)
    array([-50., -40., -30., -20., -10.,   0.,  10.,  20.,  30.,  40.,  50.])
    >>> disp = np.linspace(3000., 8000., 6)
    array([ 3000.,  4000.,  5000.,  6000.,  7000.,  8000.])
    >>> flux = np.repeat(np.array([[0.], [1.], [2.], [3.], [4.], [5.],
    ...                            [4.], [3.], [2.], [1.], [0.]]),
    ...                  6, axis=1)
    array([[ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.,  1.],
	   [ 2.,  2.,  2.,  2.,  2.,  2.],
	   [ 3.,  3.,  3.,  3.,  3.,  3.],
	   [ 4.,  4.,  4.,  4.,  4.,  4.],
	   [ 5.,  5.,  5.,  5.,  5.,  5.],
	   [ 4.,  4.,  4.,  4.,  4.,  4.],
	   [ 3.,  3.,  3.,  3.,  3.,  3.],
	   [ 2.,  2.,  2.,  2.,  2.,  2.],
	   [ 1.,  1.,  1.,  1.,  1.,  1.],
	   [ 0.,  0.,  0.,  0.,  0.,  0.]])
    >>> model = sncosmo.TimeSeriesModel(phase, disp, flux)
    >>> print model
    Model class: TimeSeriesModel
    Model name: None
    Model version: None
    Model phases: [-50, .., 50] days (11 points)
    Model dispersion: [3000, .., 8000] Angstroms (6 points) 
    Reference phase: 0 days
    Cosmology: WMAP9(H0=69.3, Om0=0.286, Ode0=0.713)
    Current Parameters:
        fscale = 1.0
        m = None [bessellb, ab]
        mabs = None [bessellb, ab]
        t0 = 0.0
        z = None
        c = None

Extinction
----------

Extinction in both models is specified by a function that accepts an
array of wavelengths in Angstroms and returns the extinction in
magnitudes for each wavelength for ``c=1``. (In other words, it should
return the *ratio* of extinction in magnitudes to the ``c``
parameter). By default, the extinction is the Cardelli, Clayton and
Mathis (CCM) law, with :math:`R_V = 3.1`. The extinction
function can be changed two ways:

1. Using the ``set_extinction_func`` method on an existing model object. This example will change the extinction to a CCM law with :math:`R_V = 2`.

    >>> model.set_extinction_func(sncosmo.extinction_ccm, extra_params={'ebv':1., r_v=2.}

2. Upon initialization of the model from data (as above), specify the
   ``extinction_func`` and ``extinction_kwargs`` parameters:

    >>> model = sncosmo.TimeSeriesModel(phase, disp, flux,
    ...                                 extinction_func=sncosmo.extinction_ccm,
    ...                                 extinction_kwargs={'ebv':1., 'r_v':2.})

Internally, the model evaluates the extinction once at the native
wavelengths of the model and stores the flux transmission values
(interpreted as corresponding to ``c=1``. When needed, the extinction
flux transmission values are calculated as ``(stored flux
transmission) ** c``. Spline interpolation is used to interpolate
between native model wavelengths.

Model Particulars: ``SALT2Model``
=================================

Initializing
------------

The SALT2 model is initialized directly from data files representing the model.
You can initialize it by giving it a path to a directory containing the files.

    >>> model = sncosmo.SALT2Model(modeldir='/path/to/dir')

By default, the initializer looks for files with names like 
``'salt2_template_0.dat'``, but this behavior can be altered with keyword
parameters:

    >>> model = sncosmo.SALT2Model(modeldir='/path/to/dir',
    ...                            m0file='mytemplate0file.dat')

See `~sncosmo.SALT2Model` for more details.

Creating New Models Classes
===========================

In this package, a "model" is something that specifies the spectral
timeseries as a function of an arbitrary number of parameters. For
example, the SALT2 model has two parameters (`x1` and `c`) that
determine a unique spectrum as a function of phase. New models can be
easily implemented by deriving from the abstract base class
`sncosmo.Model` and inheriting most of the functionality described here.
