# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""The sncosmo registry is used to load and return instances in memory
based on string identifiers."""

from collections import OrderedDict
from astropy.extern import six

__all__ = ['register_loader', 'register']

_loaders = OrderedDict()
_instances = OrderedDict()


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

    if args is None:
        args = []
    if meta is None:
        meta = {}

    # Convert name to lowercase. The registry stores names in all-lowercase.
    name = name.lower()

    # define the key
    if version is None:
        key = (data_class, name)
    else:
        key = (data_class, name, version)

    # Add the loader to the registry if it is not already there.
    if key not in _loaders or force:
        _loaders[key] = func, args, meta
    else:
        if version is not None:
            versionstr = " (version='{0:s}')".format(version)
        else:
            versionstr = ""
        raise Exception("Loader for {0:s} named '{1:s}'{2:s} is already "
                        "defined. Use force=True to override."
                        .format(data_class.__name__, name, versionstr))


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

    if name is None:
        try:
            name = instance.name
        except AttributeError:
            raise ValueError("name not given and instance has no 'name' "
                             "attribute")
        if not isinstance(name, six.string_types):
            raise ValueError("name attribute of {0!r:s} is not a string.")

    if data_class is None:
        data_class = instance.__class__

    name = name.lower()
    key = (data_class, name)

    already_present = (key in _instances) or (key in _loaders)
    if not already_present or force:
        _instances[key] = instance
    else:
        raise Exception("{0:s} named {1:s} already in registry. Use force=True"
                        " to override.".format(data_class.__name__, name))


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

    if isinstance(name, data_class):
        return name

    # Try to retrieve from the instances assuming `name` is lowercase.
    if version is None:
        key = (data_class, name)
    else:
        key = (data_class, name, version)
    try:
        return _instances[key]
    except KeyError:
        pass

    # Try to convert name to lowercase first.
    name = name.lower()
    if version is None:
        key = (data_class, name)
    else:
        key = (data_class, name, version)
    try:
        return _instances[key]
    except KeyError:
        pass

    # Try to retrieve from the loaders.
    if key in _loaders:
        func, args, meta = _loaders[key]
        if version is None:
            _instances[key] = func(*args, name=name)
        else:
            _instances[key] = func(*args, name=name, version=version)
        return _instances[key]

    # If version not specified, find the latest version and try to load it.
    if version is None:
        latest_version = None
        for regkey in _loaders.keys():
            if (key == regkey[0:2] and (latest_version is None or
                                        regkey[2] > latest_version)):
                latest_version = regkey[2]
        if latest_version is not None:
            regkey = (key[0], key[1], latest_version)
            func, args, meta = _loaders[regkey]
            _instances[regkey] = func(*args, name=name, version=latest_version)
            _instances[key] = _instances[regkey]
            return _instances[key]

    # At this point we will raise an exception.
    registered_names = [regkey[1] for regkey in _loaders.keys()
                        if regkey[0] is data_class]
    for regkey in _instances.keys():
        if regkey[0] is data_class and regkey[1] not in registered_names:
            registered_names.append(regkey[1])

    if version is None or name not in registered_names:
        raise Exception(
            "No {0} named {1!r} in registry. Registered names: '{2}'"
            .format(data_class.__name__, name, "', '".join(registered_names)))

    registered_versions = [regkey[2] for regkey in _loaders.keys()
                           if key[0:2] == regkey[0:2]]
    raise Exception(
        "No {0:s} named '{1:s}' with version='{2:s}' in registry. Registered"
        " versions: '{3:s}'".format(data_class.__name__, name, version,
                                    "', '".join(registered_versions)))


def get_loaders_metadata(data_class):
    """Return the metadata of all registered loaders for a given class.

    Parameters
    ----------
    data_class : classobj

    Returns
    -------
    loadermeta : list of dict
        Each item in the list is a dictionary containing a 'name'
        keyword, a 'version' keyword (if applicable), and the metadata
        keywords for the given loader.
    """

    loaders_metadata = []
    for lkey, loader in six.iteritems(_loaders):
        if lkey[0] is not data_class:
            continue
        m = {'name': lkey[1]}
        if len(lkey) > 2:
            m['version'] = lkey[2]
        m.update(loader[2])
        loaders_metadata.append(m)
    return loaders_metadata
