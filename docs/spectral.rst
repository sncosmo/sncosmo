*******************
Spectral operations
*******************

Bandpasses
----------

You can get a built-in bandpass using the class method `Bandpass.from_name()`:

    >>> from sncosmo import Bandpass
    >>> b = Bandpass.from_name('bessellb')

See below for a list of built-in bandpasses. In addition to string
identifiers, this class method also accepts `Bandpass` instances. That
means a function can accept either a `Bandpass` or a string
identifier, like this:

    >>> def my_function(band):
    >>>     band = Bandpass.from_name(band)
    >>>     ...

If `band` is already a `Bandpass`, it is directly returned. If it is a
string, the registry is used to return a `Bandpass` if one is
registered with that string identifier.

Built-in Bandpasses
```````````````````

.. automodule:: sncosmo._builtin.bandpasses

.. plot:: pyplots/bandpasses.py


