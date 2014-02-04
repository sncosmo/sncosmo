********
Registry
********

What is it?
-----------

The registry (`sncosmo.registry`) is responsible for translating
string identifiers to objects, for user convenience. For example, it is
used in `sncosmo.get_bandpass` and `sncosmo.get_source` to return a
`~sncosmo.Bandpass` or `sncosmo.Model` object based on the name of
the bandpass or model:

    >>> sncosmo.get_bandpass('sdssi')
    <Bandpass 'sdssi' at 0x28e7c90>

It is also used in methods like `~sncosmo.Model.bandflux` to
give it the ability to accept either a `~sncosmo.Bandpass` object or
the name of a bandpass:

    >>> model = sncosmo.Model(source='hsiao')
    >>> model.bandflux('sdssg', 0.)  # works, thanks to registry.

Under the covers, the ``bandflux`` method retrieves the `~sncosmo.Bandpass`
corresponding to ``'sdssg'`` by calling the
`sncosmo.registry.retrieve` function.

The registry is actually quite simple: it basically amounts to a
dictionary and a few functions for accessing the dictionary. Most of
the time, a user doesn't need to know anything about the
registry. However, it is useful if you want to add your own
"built-ins" or change the name of existing ones.

Using the registry to achieve custom "built-ins"
------------------------------------------------

There are a small set of "built-in" models, bandpasses, and magnitude
systems. But what if you want additional ones?

Create a file ``mydefs.py`` that registers all your custom definitions::

    # contents of mydefs.py
    import numpy as np
    import sncosmo

    wave = np.array([4000., 4200., 4400., 4600., 4800., 5000.])
    trans = np.array([0., 1., 1., 1., 1., 0.])
    band = sncosmo.Bandpass(wave, trans, name='tophatg')

    sncosmo.registry.register(band)

Make sure ``mydefs.py`` is somewhere in your ``$PYTHONPATH`` or the
directory you are running your main script from. Now in your script
import your definitions at the beginning::

    >>> import sncosmo
    >>> import mydefs
    >>> # ... proceed as normal
    >>> # you can now use 'tophatg' as a built-in

Changing the name of built-ins
------------------------------

To change the name of the ``'sdssg'`` band to ``'SDSS_G'``::

    # contents of mydefs.py
    import sncosmo

    band = sncosmo.get_bandpass('sdssg')
    band.name = 'SDSS_G'
    sncosmo.registry.register(band)


Large built-ins
---------------

What if your built-ins are really big or you have a lot of them? You
might only want to load them as they are needed, rather than having to
load everything into memory when you do ``import mydefs``. You can use
the `sncosmo.registry.register_loader` function. Suppose we have a
bandpass that requires a huge data file (In reality it is unlikely
that loading bandpasses would take a noticable amount of time, but it
might for models or spectra.)::

    # contents of mydefs.py
    import sncosmo

    def load_bandpass(filename, name=None, version=None):
        # ...
        # read data from filename, create a Bandpass object, "band"
        # ...
        return band

    filename = 'path/to/datafile/for/huge_tophatg'
    sncosmo.registry.register_loader(
        sncosmo.Bandpass,      # class of object returned.
        'huge_tophatg',        # name
        load_bandpass,         # function that does the loading
        [filename]             # arguments to pass to function
        )

Now when you ``import mydefs`` the registry will know how to load the
`~sncosmo.Bandpass` named ``'huge_tophatg'`` when it is needed. When
loaded, it will be saved in memory so that subsequent operations don't
need to load it again.
