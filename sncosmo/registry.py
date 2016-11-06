# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Public interface functions for registering and retrieving from registries"""

from . import models
from . import bandpasses
from . import magsystems

__all__ = ['register_loader', 'register']


def _get_registry(data_class):
    if issubclass(data_class, bandpasses.Bandpass):
        return bandpasses._BANDPASSES
    elif issubclass(data_class, magsystems.MagSystem):
        return magsystems._MAGSYSTEMS
    elif issubclass(data_class, models.Source):
        return models._SOURCES

    raise ValueError("No registry for type: {}".format(data_class))


def register_loader(data_class, name, func, args=None,
                    version=None, meta=None, force=False):
    """Register a data reading function.

    .. note:: Formerly accessed as ``sncosmo.registry.register_loader`` prior
              to v1.2.

    Parameters
    ----------
    data_class : classobj
        The class of the object that the loader returns.
    name : str
        The data identifier.
    func : callable
        The function to read in the data. Must accept a name and version
        keyword argument.
    args : list, optional
        Arguments to pass to the function. Default is an empty list.
    version : str, optional
        Sub-version of name, if desired. Use formats such as ``'1'``,
        ``'1.0'``, ``'1.0.0'``, etc. Default is `None`.
    force : bool, optional
        Whether to override any existing function if already present.
    meta : dict, optional
        Metadata describing this loader. Default is an empty dictionary.
    """

    _get_registry(data_class).register_loader(
        name, func, args=args, version=version, meta=meta, force=force)


def register(instance, name=None, data_class=None, force=False):
    """Register a class instance.

    .. note:: Formerly accessed as ``sncosmo.registry.register`` prior
              to v1.2.

    Parameters
    ----------
    instance : object
        The object to be registered.
    name : str, optional
        Identifier. If `None`, the name is taken from the `name` attribute
        of the instance, if it exists and is a string.
    data_class : classobj, optional
        If given, the instance is registered as an instance of this class
        rather the the class of the instance itself. Use this for registering
        subclasses when you wish them to be accessible from their superclass.
    force : bool, optional
        Whether to override any existing instance of the same name. Note: this
        may not play well with versioned instances.
    """

    if data_class is None:
        data_class = instance.__class__

    _get_registry(data_class).register(instance, name=name, force=force)


def retrieve(data_class, name, version=None):
    """Retrieve a class instance from a registered identifier.

    Parameters
    ----------
    data_class : classobj
        The class of the object requested.
    name : str
        Identifier of the specific instance. `name` is case-independent,
        however, note the following: Internally, names are stored in lowercase.
        The retrieval is first tried assuming the requested name is also
        lowercase, then `name` is converted to lowercase. So it should be
        slightly faster to use lowercase names everywhere.
    version : str
        Sub-identifier. If `None`, default to highest or only version.

    Returns
    -------
    instance : data_class (or subclass thereof)

    Notes
    -----
    **Precedence** The following are tried in this order:

    1. If `name` is already an instance of `data_class`
       (rather than a string), it is immediately returned.
    2. If (`data_class`, `name`) is already in the registry,
       that instance is returned.
    3. If there is a loader defined for (`data_class`, `name`), it is used
       to create an instance, save it to the registry and return it.
    4. An Exception is raised listing the available registered names for
       the requested data class.

    **Versioning** There is support for multiple versions of data for the
    same `name`.

    1. If `version` is specified, the registry and its loaders are
       searched for (`data_class`, `name`, `version`).
    2. If `version` is not specified but there are registered loaders for
       (`data_class`, `name`, `version`), the latest version is used,
       and both (`data_class`, `name`) and (`data_class`, `name`,
       `version`) are saved to the registry. "Latest" is defined by
       string comparision.

    """

    _get_registry(data_class).retrieve(name, version=version)
