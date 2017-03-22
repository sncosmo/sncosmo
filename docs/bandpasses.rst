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

    >>> import sncosmo
    >>> band = sncosmo.get_bandpass('sdssi')
    >>> band
    <Bandpass 'sdssi' at 0x...>

To create a Bandpass directly, you can supply arrays of wavelength and
transmission values:

    >>> wavelength = [4200., 4400., 4600., 4800.]
    >>> transmission = [1., 1., 1., 1.]
    >>> sncosmo.Bandpass(wavelength, transmission, name='tophatg')
    <Bandpass 'tophatg' at 0x...>

By default, the first argument is assumed to be wavelength in Angstroms.
To specify a different dispersion unit, use a unit from the
`astropy.units` package:

    >>> import astropy.units as u
    >>> wavelength = [420., 440., 460., 480.]
    >>> transmission = [1., 1., 1., 1.]
    >>> Bandpass(wavelength, transmission, wave_unit=u.nm)
    <Bandpass 'tophatg' at 0x...>

Using a Bandpass
----------------

A Bandpass acts like a continuous 1-d function, returning the transmission
at supplied wavelengths (always in Angstroms)::

    >>> band([4100., 4250., 4300.])
    array([ 0.,  1.,  1.])

Note that the transmission is zero outside the defined wavelength range.
Linear interpolation is used between the defined wavelengths.

Bnadpasses have a few other useful properties. You can get the range of
wavelengths where the transmission is non-zero::

    >>> band.minwave(), band.maxwave()
    (4200.0, 4800.0)

Or the transmission-weighted effective wavelength:

    >>> band.wave_eff
    4500.0

Or the name:

    >>> band.name
    'tophatg'


Adding Bandpasses to the Registry
---------------------------------

You can create your own bandpasses and use them like built-ins by adding them
to the registry. Suppose we want to register the 'tophatg' bandpass we created:

    >>> sncosmo.register(band, 'tophatg')

Or if ``band.name`` has been set:

    >>> sncosmo.register(band)  # registers band under band.name

After doing this, we can get the bandpass object by doing

    >>> band = sncosmo.get_bandpass('tophatg')

Also, **we can pass the string** ``'tophatg'`` **to any function that
takes a** `~sncosmo.Bandpass` **object**. This means that you can create
and register bandpasses at the top of a script, then just keep track
of string identifiers throughout the rest of the script.
