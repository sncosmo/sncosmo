# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Something about the registry."""

from astropy.utils import OrderedDict

__all__ = ['register_loader', 'retrieve', 'get_loaders_metadata']

_loaders = OrderedDict()
_instances = OrderedDict()

def register_loader(data_class, name, function, function_args=None,
                    version=None, force=False, **meta):
    """Register a data reading function.

    Parameters
    ----------
    data_class : classobj
        The class of the object that the loader returns.
    name : str
        The data identifier.
    function : function
        The function to read in the data.
    function_args : list, optional
        Arguments to pass to the function. Default is an empty list.
    version : str, optional
        Sub-version of name, if desired. Use formats such as ``'1'``,
        ``'1.0'``, ``'1.0.0'``, etc. Default is `None`. 
    force : bool, optional
        Whether to override any existing function if already present.
    **kwargs : optional
        Any additional keyword arguments are assumed to be metadata and
        are saved as a dictionary describing this loader.
    """

    if function_args is None:
        function_args = []

    if version is None:
        key = (data_class, name)
    else:
        key = (data_class, name, version)

    if not key in _loaders or force:
        _loaders[key] = function, function_args, meta
    else:
        if version is not None:
            versionstr = " (version='{0:s}')".format(version)
        else:
            versionstr = ""
        raise Exception("Loader for {0:s} named '{1:s}'{2:s} is already "
                        "defined. Use force=True to override."
                        .format(data_class.__name__, name, versionstr))


def retrieve(data_class, name, version=None):
    """Retrieve a class instance from a registered identifier.

    Parameters
    ----------
    data_class : classobj
        The class of the object requested.
    name : str
        Identifier of the specific instance.
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

    if isinstance(name, data_class): return name
    if version is None:
        key = (data_class, name)
    else:
        key = (data_class, name, version)

    try:
        return _instances[key]
    except KeyError:
        pass

    if key in _loaders:
        function, function_args, meta = _loaders[key]
        _instances[key] = function(*function_args)
        return _instances[key]

    if version is None:
        latest_version = None
        for regkey in _loaders.keys():
            if (key == regkey[0:2] and
                (regkey[2] > latest_version or latest_version is None)):
                latest_version = regkey[2]
        if latest_version is not None:
            regkey = (key[0], key[1], latest_version)
            function, function_args, meta = _loaders[regkey]
            _instances[regkey] = function(*function_args)
            _instances[key] = _instances[regkey]
            return _instances[key]

        
    registered_names = [regkey[1] for regkey in _loaders.keys()
                        if regkey[0] is data_class]

    if version is None or name not in registered_names:
        raise Exception(
            "No {0:s} named '{1:s}' in registry. Registered names: '{2:s}'"
            .format(data_class.__name__, name, "', '".join(registered_names)))

    registered_versions = [
        regkey[2] for regkey in _loaders.keys() if key[0:2] == regkey[0:2]]
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
    loadermeta : list
        Each item in the list is a dictionary containing a 'name'
        keyword, a 'version' keyword (if applicable), and the metadata
        keywords for the given loader.
    """

    loaders_metadata = []
    for lkey, loader in _loaders.iteritems():
        if lkey[0] is not data_class: continue
        m = {'name': lkey[1]}
        if len(lkey) > 2: m['version'] = lkey[2]
        m.update(loader[2])
        loaders_metadata.append(m)
    return loaders_metadata
