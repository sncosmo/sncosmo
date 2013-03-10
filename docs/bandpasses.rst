**********
Bandpasses
**********

Constructing a Bandpass
-----------------------

Bandpass objects represent the transmission fraction of an
astronomical filter as a function of dispersion (photon wavelength,
frequency or energy). To create a Bandpass:

    >>> import numpy as np
    >>> from sncosmo import Bandpass
    >>> dispersion = np.array([4000., 4200., 4400., 4600., 4800., 5000.])
    >>> trans = np.array([0., 1., 1., 1., 1., 0.])
    >>> band = Bandpass(dispersion, trans)

By default, the dispersion is assumed to be wavelengths with units of
Angstroms. It must be monotonically increasing. To specify a different
dispersion unit, use a unit from the `astropy.units` package:

    >>> import astropy.units as u
    >>> dispersion = np.array([400., 420., 440., 460., 480., 500.])
    >>> band = Bandpass(dispersion, trans, dunit=u.nm)

You can specify a name upon construction, and the bandpass will save it
internally:

    >>> band = Bandpass(dispersion, trans, name='tophatg')
    >>> band
    <Bandpass 'tophatg' at 0x37869d0>

It can also be set using the `name` attribute:

    >>> band.name = 'tophatg'
    >>> band.name
    'tophatg'

Using a Bandpass
----------------

Once created, you can access the values:

    >>> band.dispersion
    array([ 4000.,  4200.,  4400.,  4600.,  4800.,  5000.])
    >>> band.transmission
    array([ 0.,  1.,  1.,  1.,  1.,  0.])
    >>> band.dunit
    Unit("Angstrom")

To convert the dispersion to a different unit:

    >>> band2 = band.to_unit(u.Hz)
    >>> band2
    <Bandpass 'tophatg' at 0x39d2c50>
    >>> band2.dispersion
    array([  5.99584916e+14,   6.24567621e+14,   6.51722735e+14,
             6.81346495e+14,   7.13791567e+14,   7.49481145e+14])
    >>> band2.transmission
    array([ 0.,  1.,  1.,  1.,  1.,  0.])
    >>> band2.dunit
    Unit("Hz")

This generally creates a new bandpass object, unless the requested
units are the same as the current units:

    >>> band2 = band.to_unit(u.AA)
    >>> band
    <Bandpass 'tophatg' at 0x37869d0>
    >>> band2
    <Bandpass 'tophatg' at 0x37869d0>
    >>> band2 is band
    True


Built-in Bandpasses
-------------------

You can get a built-in bandpass using the class method
`Bandpass.from_name()`:

    >>> band = Bandpass.from_name('bessellb')

If there are no built-in bandpass with that name, an exception is raised.
See the :ref:`list-of-built-in-bandpasses` below.

Using Bandpass names in place of Bandpasses
-------------------------------------------

The functions in `sncosmo` that require a `Bandpass` can also accept
the *name* of a built-in bandpass in place of an actual `Bandpass`
object.  This saves you from having to create the bandpass directly.
Internally, these functions call the above-mentioned method
`from_name`, like this:

    >>> def some_function(band):
    >>>     band = Bandpass.from_name(band)
    >>>     ... use band ...

If `band` is a `Bandpass`, it is directly returned. If `band` is a
string, the corresponding built-in `Bandpass` is returned. 

Adding Bandpasses to the Registry
---------------------------------

You can create your own bandpasses and use them like built-ins by adding them
to the registry. Suppose we want to register the 'tophatg' bandpass we created
above:

    >>> from sncosmo import registry
    >>> registry.register(band, 'tophatg')

After doing this, we can pass the string 'tophatg' to any function that
takes a `Bandpass` object. If the `name` attribute of band is defined and is
a string, we can simply do

    >>> registry.register(band)

and the band will be registered under `band.name`.

.. _list-of-built-in-bandpasses:

List of Built-in Bandpasses
---------------------------

.. automodule:: sncosmo._builtin.bandpasses

.. plot:: pyplots/bandpasses.py


