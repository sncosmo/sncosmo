**********
Bandpasses
**********

Constructing a Bandpass
-----------------------

Bandpass objects represent the transmission fraction of an
astronomical filter as a function of dispersion (photon wavelength,
frequency or energy). They are basically simple containers for arrays of these
values, with a couple special features. To get a bandpass that is in
the registry (built-in)::

    >>> import sncosmo  # doctest: +ELLIPSIS
    >>> band = sncosmo.get_bandpass('sdssi')
    >>> band
    <Bandpass 'sdssi' at 0x...>

If you try to retrieve a bandpass that is not in the registry, you
will get a list of bandpasses that *are* in the registry::

    >>> band = sncosmo.get_bandpass('perfect_tophat')
    Traceback (most recent call last):
    ...
    Exception: No Bandpass named 'perfect_tophat' in registry. Registered names: 'desg', 'desr', ..., 'sdssi', 'sdssz'

See the :ref:`list-of-built-in-bandpasses`.

To create a Bandpass directly:

    >>> import numpy as np
    >>> wavelength = np.array([4000., 4200., 4400., 4600., 4800., 5000.])
    >>> transmission = np.array([0., 1., 1., 1., 1., 0.])
    >>> sncosmo.Bandpass(wavelength, transmission, name='tophatg')
    <Bandpass 'tophatg' at 0x...>

By default, the dispersion is assumed to be wavelengths with units of
Angstroms. It must be monotonically increasing. To specify a different
dispersion unit, use a unit from the `astropy.units` package:

    >>> import astropy.units as u
    >>> wavelength = np.array([400., 420., 440., 460., 480., 500.])
    >>> transmission = np.array([0., 1., 1., 1., 1., 0.])
    >>> band = Bandpass(wavelength, transmission, wave_unit=u.nm)

Outside the defined dispersion range, the transmission is treated as being 0. 

Using a Bandpass
----------------

Once created, you can access the values:

    >>> band.wave  # always returns wavelengths in Angstroms
    array([ 4000.,  4200.,  4400.,  4600.,  4800.,  5000.])
    >>> band.trans
    array([ 0.,  1.,  1.,  1.,  1.,  0.])
    >>> band.name
    'tophatg'
    >>> band.dwave  # width of each "bin" in Angstroms
    array([ 200.,  200.,  200.,  200.,  200.,  200.])
    >>> band.wave_eff  # effective wavelength (transmission-weighted)
    4500.0


Adding Bandpasses to the Registry
---------------------------------

You can create your own bandpasses and use them like built-ins by adding them
to the registry. Suppose we want to register the 'tophatg' bandpass we created:

    >>> sncosmo.registry.register(band, 'tophatg')

Or if ``band.name`` has been set:

    >>> sncosmo.registry.register(band)  # registers band under band.name

After doing this, we can get the bandpass object by doing

    >>> band = sncosmo.get_bandpass('tophatg')

Also, *we can pass the string* ``'tophatg'`` *to any function that
takes a* `~sncosmo.Bandpass` *object*. This means that you can create
and register bandpasses at the top of a script, then just keep track
of string identifiers throughout the rest of the script.
