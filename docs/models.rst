****************
Supernova Models
****************

Initializing from built-in models
=================================

Create an `~sncosmo.ObsModel` that has the SALT2 model as a source:

    >>> import sncosmo
    >>> model = sncosmo.ObsModel(source='salt2')
    >>> print model
    <ObsModel at 0x48585d0>
    source:
      class      : SALT2Model
      name       : salt2
      version    : 2.0
      phases     : [-20, .., 50] days (71 points)
      wavelengths: [2000, .., 9200] Angstroms (721 points)
    parameters:
      z  = 0.0
      t0 = 0.0
      x0 = 1.0
      x1 = 0.0
      c  = 0.0

.. note:: In fact, the data for "built-in" models like SALT2 are hosted
	  remotely, downloaded as needed, and cached locally. So the first
	  time you load a given model, you need
	  to be connected to the internet.  You will see a progress bar as
          the data are downloaded.

Some source models have multiple versions of the data, which can be explicitly
retrieved using the `~sncosmo.get_sourcemodel` function:

    >>> source = sncosmo.get_sourcemodel('salt2', version='1.1')
    >>> model = sncosmo.ObsModel(source=source)


Retrieving and setting parameters
=================================

Each model has a set of parameter names, that can be retrieved via:

    >>> model.param_names
    ['z', 't0', 'x0', 'x1', 'c']

Each *instance* also has a set of parameter values, corresponding to the
parameter names:

    >>> model.parameters
    array([ 0.,  0.,  1.,  0.,  0.])

These can also be retrieved as:

    >>> model['z']  # Note: this syntax is preliminary!
    0.0

Parameter values can be set by (1) explicitly indexing the parameter array
or (2) by using the "dictionary"-like syntax or (3) using the ``set`` method:

    >>> model.parameters[0] = 0.5
    >>> model['z'] = 0.5  # Note: this syntax is preliminary!
    >>> model.set(z=0.5)

The set method can take multiple parameters.

What do these parameters mean? The first two, ``z`` and ``t0`` are common to
all `~sncosmo.ObsModel` instances. The next three, ``x0``, ``x1`` and ``c``
are specific to the particular source, in this case the SALT2 model. Other
source models might have different parameters.

* ``z`` is the redshift of the source.
* ``t0`` is the observer-frame time corresponding to the model's phase=0.

Adding effects to the model
===========================

TODO: Write this.


Native model phases and wavelengths
===================================

The model is defined at discrete phases and wavelengths and interpolation is
used to determine flux values at intermediate phase(s) and wavelength(s). To
see what the native values are:

    >>> model.times
    array([ -36. ,  -34.2,  -32.4, ..., 149.4, 151.2,  153. ])
    >>> model.wave
    array([  1500.,   1515.,   1530., ...,  37470.,  37485.,  37500.])

.. warning::
   Everything below is from the old docs and hasn't been updated yet.


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
