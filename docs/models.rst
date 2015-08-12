****************
Supernova Models
****************

Getting Started
===============

Create a model using the built-in "source" named ``'hsiao'``:

    >>> import sncosmo
    >>> model = sncosmo.Model(source='hsiao')

Set the redshift, time-of-zero-phase and the amplitude:

    >>> model.set(z=0.5, t0=55000., amplitude=1.e-10)

Generate synthetic photometry through an observer-frame bandpass:

    >>> model.bandmag('desr', 'ab', [54990., 55000., 55020.])
    array([ 24.82381795,  24.41496701,  25.2950865 ])

Equivalent values in photons / s / cm^2:

    >>> model.bandflux('desr', [54990., 55000., 55020.])
    array([  7.22413301e-05,   1.05275209e-04,   4.68034980e-05])

Equivalent values scaled so that 1 is equivalent to an AB magnitude of 25:

    >>> model.bandflux('desr', [54990., 55000., 55020.], zp=25., zpsys='ab')
    array([ 1.17617737,  1.71400939,  0.7620183 ])

Generate an observer-frame spectrum at a given time and wavelengths
(in ergs/s/cm^2/Angstrom):

    >>> model.flux(54990., [4000., 4100., 4200.])
    array([  4.31210900e-20,   7.46619962e-20,   1.42182787e-19])


Creating a model using a built-in source
========================================

A Model in sncosmo consists of

* **One "source"** A model of the spectral evolution of the source
  (e.g., a supernova).
* **Zero or more "propagation effects"** Models of how intervening structures
  (e.g., host galaxy dust, milky way dust) affect the spectrum.

In the above example, we created a model with no propagation effects,
using one of the built-in ``Source`` instances that sncosmo knows
about: ``'hsiao'``. See the full :ref:`list-of-built-in-sources` that
sncosmo knows about.

.. note:: In fact, the data for "built-in" sources are hosted
	  remotely, downloaded as needed, and cached locally. So the
	  first time you load a given model, you need to be connected
	  to the internet.  You will see a progress bar as the data
	  are downloaded.  By default, SNCosmo will use a subdirectory
	  of the AstroPy cache directory for this purpose, e.g.,
	  ``$HOME/.astropy/cache/sncosmo``, but this can be changed
	  by setting the ``data_dir`` configuration parameter in 
          ``$HOME/.astropy/config/sncosmo.cfg``.  See :doc:`configuration`
	  for more information.

Some built-in source models have multiple versions, which can
be explicitly retrieved using the `~sncosmo.get_source` function:

    >>> source = sncosmo.get_source('hsiao', version='2.0')
    >>> model = sncosmo.Model(source=source)

Model parameters
================

Each model has a set of parameter names and values:

    >>> model.param_names
    ['z', 't0', 'amplitude']
    >>> model.parameters
    array([ 0.,  0.,  1.])

These can also be retrieved as:

    >>> model.get('z')
    0.0

Parameter values can be set by explicitly indexing the parameter array
or by using the ``set`` method:

    >>> model.parameters[0] = 0.5
    >>> model.set(z=0.5)
    >>> model.set(z=0.5, amplitude=2.0)  # Can specify multiple parameters

What do these parameters mean? The first two, ``z`` and ``t0`` are
common to all `~sncosmo.Model` instances:

* ``z`` is the redshift of the source.
* ``t0`` is the observer-frame time corresponding to the source's phase=0.

Note that in some sources phase=0 might be at explosion while others
might be at max: the definition of phase is arbitrary. However,
observed time is always related to phase via ``time = t0 + phase *
(1 + z)``

The next, ``amplitude``, is specific to the particular type of
source. In this case, the source is a simple spectral timeseries that
can only be scaled up and down. Other sources could have other
parameters that affect the shape of the spectrum at each phase.

For a given model, you can set the ``amplitude`` (or ``x0`` in case you
are using a SALT model) according to a desired absolute magnitude in a
specific band by using the method
`~sncosmo.Model.set_source_peakabsmag()`. Note that the redshift ``z`` affects
your result. Therefore, you could specify:

     >>> model.set(z=1.6)
     >>> model.set_source_peakabsmag(-19.0, 'bessellb', 'ab')

Specifically, for SALT models, it is recommended to call
`~sncosmo.Model.set_source_peakabsmag()` after setting the other model
parameters, such as ``x1`` and ``c``. It probably won't make a
difference if you are using the ``'bessellb'`` bandpass, but if you
were setting the absolute magnitude in another band, it would make a
small difference.

The reason for this peculiarity is that "absolute magnitude" is not a
parameter in the SALT2 model, per se. The parameters are ``x0``, ``x1``,
``c``, ``t0``, and ``z``. ``x0`` is a simple multiplicative scaling factor on
the whole spectral timeseries. The ``set_source_peakabsmag()`` method is a
convenience for setting ``x0`` such that the integrated flux through a
given bandpass is as desired. Since the integrated flux depends on the
spectral shape, it will depend on ``x1`` and ``c``.

Creating a model with a source and effect(s)
============================================

Let's create a slightly more complex model. Again we will use the Hsiao
spectral time series as a source, but this time we will add host galaxy
dust.

    >>> dust = sncosmo.CCM89Dust()
    >>> model = sncosmo.Model(source='hsiao',
    ...                       effects=[dust],
    ...                       effect_names=['host'],
    ...                       effect_frames=['rest'])

The model now has additional parameters that describe the dust, ``hostebv``
and ``hostr_v``:

    >>> model.param_names
    ['z', 't0', 'amplitude', 'hostebv', 'hostr_v']
    >>> model.parameters
    array([ 0. ,  0. ,  1. ,  0. ,  3.1])

These are the parameters of the ``CCM89Dust`` instance we created:

    >>> dust.param_names
    ['ebv', 'r_v']

In the model, the parameter names are prefixed with the name of the effect
(``host``).

At any time you can print the model to get a nicely formatted string
representation of its components and current parameter values:

    >>> print model
    <Model at 0x...>
    source:
      class      : TimeSeriesSource
      name       : hsiao
      version    : 3.0
      phases     : [-20, .., 85] days (22 points)
      wavelengths: [1000, .., 25000] Angstroms (481 points)
    effect (name='host' frame='rest'):
      class           : CCM89Dust
      wavelength range: [1250, 33333] Angstroms
    parameters:
      z         = 0.0
      t0        = 0.0
      amplitude = 1.0
      hostebv   = 0.0
      hostr_v   = 3.1000000000000001

Also, ``str(model)`` will return this string rather than printing it.


Adding Milky Way dust
=====================

Dust in the Milky Way will affect the shape of an observed supernova
spectrum.  It is important to take this into account in our model when
fitting the model to observed data.  As with host galaxy dust treated
above, we can model Milky Way dust as a "propagation effect". The only
difference is that Milky Way dust is in the observer frame rather than the
supernova rest frame. Here, we create a model with dust in *both* the
SN rest frame and the observer frame::

    >>> dust = sncosmo.CCM89Dust()
    >>> model = sncosmo.Model(source='hsiao',
    ...                       effects=[dust, dust],
    ...                       effect_names=['host', 'mw'],
    ...                       effect_frames=['rest', 'obs'])

We can see that the model includes four extra parameters (two describing the
host galaxy dust and two describing the milky way dust)::

    >>> model.param_names
    ['z', 't0', 'amplitude', 'hostebv', 'hostr_v', 'mwebv', 'mwr_v']
    >>> model.parameters  # default values
    array([ 0. ,  0. ,  1. ,  0. ,  3.1,  0. ,  3.1])

The host galaxy dust parameters are prefixed with ``'host'`` and the
Milky Way dust parameters are prefixed with ``'mw'``. These are just
the names we supplied when constructing the model. The effect names
have no significance beyond this.  The effect frames, on the other
hand, *are* significant. The only allowed values are ``'rest'`` (rest
frame) and ``'obs'`` (observer frame).

A typical use pattern is to get an estimate of the amount of Milky Way
dust at the location of the supernova from a dust map, and then to fix
that amount of dust in the model.  The following example illustrates
how to do this using the Schlegel, Finkbeiner and Davis (1998) dust map.
First, load the dust map (do this only once)::

    >>> dustmap = sncosmo.SFD98Map("/path/to/dust/maps")

.. note::

   This supposes that you've downloaded the full resolution E(B-V)
   maps from `this website
   <http://www.astro.princeton.edu/~schlegel/dust/data/data.html>`_
   and placed them in the given directory
   ``"/path/to/dust/maps"``. The directory can also be set in the
   sncosmo configuration file, in which case you can just do
   ``sncosmo.SFD98Map()``. See `~sncosmo.SFD98Map` for more details.

Now, for each SN you wish to fit, get the amount of dust at the SN location
and set the ``mwebv`` model parameter appropriately. For example, if the SN is
located at RA=42.8 degrees, Dec=0 degrees::

  >>> ebv = dustmap.get_ebv((42.8, 0.))
  >>> model.set(mwebv=ebv)
  >>> # proceed with fitting the other model parameters to the data.

Note that we wish to *fix* the ``mwebv`` model parameter rather than
fitting it to the data like the other parameters: We're supposing that
this value is perfectly known from the dust map. Therefore, when using
a function such as `~sncosmo.fit_lc` to fit the parameters, be sure *not* to
include ``'mwebv'`` in the list of parameters to vary.

Model spectrum
==============

To retrieve a spectrum (in ergs / s / cm^2 / Angstrom) at a given
observer-frame time and set of wavelengths:

    >>> wave = np.array([3000., 3500., 4000., 4500., 5000., 5500.])
    >>> model.flux(-5., wave)
    array([  5.29779465e-09,   7.77702880e-09,   7.13309678e-09,
             5.68369041e-09,   3.06860759e-09,   2.59024291e-09])

We can supply a list or array of times and get a 2-d array back,
representing the spectrum at each time:

    >>> model.flux([-5., 2.], wave)
    array([[  5.29779465e-09,   7.77702880e-09,   7.13309678e-09,
              5.68369041e-09,   3.06860759e-09,   2.59024291e-09],
           [  2.88166481e-09,   6.15186858e-09,   7.87880448e-09,
              6.93919846e-09,   3.59077596e-09,   3.27623932e-09]])

Changing the model parameters changes the results:

    >>> model.parameters
    array([0., 0., 1., 0., 3.1])
    >>> model.flux(-5., [4000., 4500.])
    array([  7.13309678e-09,   5.68369041e-09])
    >>> model.set(amplitude=2., hostebv=0.1)
    >>> model.flux(-5., [4000., 4500.])
    array([  9.39081327e-09,   7.86972003e-09])


Synthetic photometry
====================

To integrate the spectrum through a bandpass, use the bandflux method:

    >>> model.bandflux('sdssi', -5.)
    180213.72886169454

Here we are using the SDSS I band, at time -5. days. The return value is in
photons / s / cm^2. It is also possible to supply multiple times or bands:

    >>> model.bandflux('sdssi', [-5., 2.])
    array([ 180213.72886169,  176662.68287381])
    >>> model.bandflux(['sdssi', 'sdssz'], [-5., -5.])
    array([ 180213.72886169,   27697.76705621])

Instead of returning flux in photons / s / cm^2, the flux can be normalized
to a desired zeropoint by specifying the ``zp`` and ``zpsys`` keywords,
which can also be scalars, lists, or arrays.

    >>> model.bandflux(['sdssi', 'sdssz'], [-5., -5.], zp=25., zpsys='ab')
    array([  5.01036850e+09,   4.74414435e+09])

Instead of flux, magnitude can be returned. It works very similarly to flux:

    >>> model.bandmag('sdssi', 'ab', [0., 1.])
    array([ 22.6255077 ,  22.62566363])
    >>> model.bandmag('sdssi', 'vega', [0., 1.])
    array([ 22.26843273,  22.26858865])

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


Initializing Sources directly
=============================

You can initialize a source directly from your own model rather than
using the built-in model data.

Initializing a ``TimeSeriesSource``
-----------------------------------

These sources are created directly from numpy arrays. Below, we build a
very simple model, of a source with a flat spectrum at all times,
rising from phase -50 to 0, then declining from phase 0 to +50.

    >>> import numpy as np
    >>> phase = np.linspace(-50., 50., 11)
    >>> disp = np.linspace(3000., 8000., 6)
    >>> flux = np.repeat(np.array([[0.], [1.], [2.], [3.], [4.], [5.],
    ...                            [4.], [3.], [2.], [1.], [0.]]),
    ...                  6, axis=1)
    >>> model = sncosmo.TimeSeriesSource(phase, disp, flux)


Initializing a ``SALT2Source``
------------------------------

The SALT2 model is initialized directly from data files representing the model.
You can initialize it by giving it a path to a directory containing the files.

    >>> model = sncosmo.SALT2Source(modeldir='/path/to/dir')

By default, the initializer looks for files with names like 
``'salt2_template_0.dat'``, but this behavior can be altered with keyword
parameters:

    >>> model = sncosmo.SALT2Source(modeldir='/path/to/dir',
    ...                             m0file='mytemplate0file.dat')

See `~sncosmo.SALT2Source` for more details.

Creating New Source Classes
===========================

A "source" is something that specifies the spectral
timeseries as a function of an arbitrary number of parameters. For
example, the SALT2 model has three parameters (``x0``, ``x1`` and ``c``) that
determine a unique spectrum as a function of phase. New models can be
easily implemented by deriving from the abstract base class
`sncosmo.Source` and inheriting most of the functionality described here.
