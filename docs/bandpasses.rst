**********
Bandpasses
**********

Constructing a Bandpass
-----------------------

Bandpass objects represent the transmission fraction of an
astronomical filter as a function of dispersion (photon wavelength,
frequency or energy). They are basically simple containers for arrays of these
values, with a couple special features. To get a bandpass that is in
the registry (built-in):

    >>> band = sncosmo.get_bandpass('sdssi')
    <Bandpass 'sdssi' at 0x28e7c90>

If you try to retrieve a bandpass that is not in the registry, you
will get a list of bandpasses that *are* in the registry:

    >>> band = sncosmo.get_bandpass('perfect_tophat')
    Exception: No Bandpass named 'perfect_tophat' in registry. Registered names: 'desg', 'desr', 'desi', 'desz', 'desy', 'bessellux', 'bessellb', 'bessellv', 'bessellr', 'besselli', 'sdssu', 'sdssg', 'sdssr', 'sdssi', 'sdssz'

See the :ref:`list-of-built-in-bandpasses`.

To create a Bandpass directly:

    >>> import numpy as np
    >>> dispersion = np.array([4000., 4200., 4400., 4600., 4800., 5000.])
    >>> trans = np.array([0., 1., 1., 1., 1., 0.])
    >>> band = sncosmo.Bandpass(dispersion, trans, name='tophatg')
    >>> band
    <Bandpass 'tophatg' at 0x37869d0>

By default, the dispersion is assumed to be wavelengths with units of
Angstroms. It must be monotonically increasing. To specify a different
dispersion unit, use a unit from the `astropy.units` package:

    >>> import astropy.units as u
    >>> dispersion = np.array([400., 420., 440., 460., 480., 500.])
    >>> trans = np.array([0., 1., 1., 1., 1., 0.])
    >>> band = Bandpass(dispersion, trans, dunit=u.nm)

Outside the defined dispersion range, the transmission is generally treated as being 0 in `sncosmo`. 

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

To convert the dispersion and transmission to a different unit:

    >>> new_disp, new_trans = band.to_unit('Hz')
    >>> new_disp
    array([  5.99584916e+14,   6.24567621e+14,   6.51722735e+14,
             6.81346495e+14,   7.13791567e+14,   7.49481145e+14])
    >>> new_trans
    array([ 0.,  1.,  1.,  1.,  1.,  0.]))


Adding Bandpasses to the Registry
---------------------------------

You can create your own bandpasses and use them like built-ins by adding them
to the registry. Suppose we want to register the 'tophatg' bandpass we created
above:

    >>> from sncosmo import registry
    >>> registry.register(band, 'tophatg')
    >>> # or if band.name is set, you can just do:
    >>> registry.register(band)  # registers band under band.name

After doing this, we can get the bandpass object by doing

    >>> band = sncosmo.get_bandpass('tophatg')

Also, *we can pass the string* ``'tophatg'`` *to any function that
takes a* `~sncosmo.Bandpass` *object*:

    >>> model = sncosmo.get_model('hsiao')
    >>> model.bandflux('tophatg')

This means that you can create and register bandpasses at the top of a
script, then just keep track of string identifiers throughout the rest
of the script.
