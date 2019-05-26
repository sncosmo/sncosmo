# Licensed under a 3-clause BSD style license - see LICENSE.rst

from collections import OrderedDict


class Registry(object):
    """Connect strings to instances and loaders."""

    def __init__(self):
        self._loaders = OrderedDict()
        self._instances = OrderedDict()
        self._primary_loaders = []  # keys of _loaders not including aliases

    def register_loader(self, name, func, args=None, version=None, meta=None,
                        force=False):
        """Register a data reading function.

        Parameters
        ----------
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

        name = name.lower()  # names are stored in all-lowercase.
        key = (name, version)

        # check if key already exists in the registry
        if key in self._loaders and not force:
            versionstr = \
                "" if version is None else " (version={!r})".format(version)
            raise Exception("Loader named {!r}{:s} is already "
                            "defined. Use force=True to override."
                            .format(name, versionstr))

        self._loaders[key] = func, args, meta
        self._primary_loaders.append(key)

    def register(self, instance, name=None, force=False):
        """Register a class instance.

        Parameters
        ----------
        instance : object
            The object to be registered.
        name : str, optional
            Identifier. If `None`, the name is taken from the `name` attribute
            of the instance, if it exists and is a string.
        force : bool, optional
            Whether to override any existing instance of the same name.
            Note: this may not play well with versioned instances.
        """

        if name is None:
            try:
                name = instance.name
            except AttributeError:
                raise ValueError("name not given and instance has no 'name' "
                                 "attribute")
            if not isinstance(name, str):
                raise ValueError("name attribute of {0!r:s} is not a string.")

        name = name.lower()
        key = (name, None)

        # check if key is already in instances or loaders:
        if (key in self._instances or key in self._loaders) and not force:
            raise Exception("{0:s} already in registry. Use force=True"
                            " to override.".format(name))

        self._instances[key] = instance

    def alias(self, new_name, existing_name, new_version=None,
              existing_version=None):
        """Alias a new name to an existing name."""

        found = False

        new_key = (new_name, new_version)
        existing_key = (existing_name, existing_version)

        if existing_key in self._loaders:
            found = True
            self._loaders[new_key] = self._loaders[existing_key]

        if existing_key in self._instances:
            found = True
            self._instances[new_key] = self._instances[existing_key]

        if not found:
            raise Exception("{0!r} not found in registry"
                            .format(existing_name))

    def retrieve(self, name, version=None):
        """Retrieve an instance from a registered identifier.

        Parameters
        ----------
        name : str
            Identifier of the specific instance. `name` is case-independent
            (internally names are stored as lowercase).
        version : str
            Sub-identifier. If `None`, default to highest or only version.

        Returns
        -------
        instance

        Notes
        -----
        **Precedence** The following are tried in this order:

        1. If ``name`` is already loaded in the registry, that instance is
           returned.
        2. If there is a loader defined for ``name``, it is used
           to create an instance, save it to the registry and return it.
        3. An Exception is raised listing the available registered names.

        **Versioning** There is support for multiple versions of data for the
        same `name`.

        1. If ``version`` is specified, the registry and its loaders are
           searched for ``(name, version)``.
        2. If ``version`` is not specified but there are registered loaders for
           ``(name, version)``, the latest version is used, and both
           ``(name, None)`` and ``(name, version)`` are saved to the registry.
           "Latest" is defined by string comparision.

        """

        name = name.lower()
        key = (name, version)

        # Try to retrieve from instances
        try:
            return self._instances[key]
        except KeyError:
            pass

        # Try to retrieve from the loaders.
        if key in self._loaders:
            func, args, meta = self._loaders[key]
            if version is None:
                self._instances[key] = func(*args, name=name)
            else:
                self._instances[key] = func(*args, name=name, version=version)
            return self._instances[key]

        # If we got this far and the version is not specified,
        # find the latest version and try to load it.
        if version is None:
            latest_version = None
            for regkey in self._loaders.keys():
                if (name == regkey[0] and (latest_version is None or
                                           regkey[1] > latest_version)):
                    latest_version = regkey[1]
            if latest_version is not None:
                regkey = (name, latest_version)
                func, args, meta = self._loaders[regkey]
                self._instances[regkey] = func(*args, name=name,
                                               version=latest_version)
                self._instances[key] = self._instances[regkey]
                return self._instances[key]

        # At this point we will raise an exception and all the following
        # is to make it more informative.

        # all names regardless of version:
        registered_names = set(k[0] for k in self._loaders)
        registered_names.update(set(k[0] for k in self._instances))

        # If we don't have the name regardless of version:
        if version is None or name not in registered_names:
            raise Exception(
                "{0!r} not in registry. Registered names: '{1}'"
                .format(name, "', '".join(registered_names)))

        # If version was specified but we don't have that specific version:
        registered_versions = [regkey[1] for regkey in self._loaders
                               if name == regkey[0]]
        raise Exception(
            "No {0!r} with version='{1:s}' in registry. Registered"
            " versions: '{2:s}'".format(name, version,
                                        "', '".join(registered_versions)))

    def get_loaders_metadata(self):
        """Return the metadata of all registered loaders.

        Returns
        -------
        loadermeta : list of dict
            Each item in the list is a dictionary containing a 'name'
            keyword, a 'version' keyword (if applicable), and the metadata
            keywords for the given loader.
        """

        result = []
        for key in self._primary_loaders:
            loader = self._loaders[key]
            name, version = key
            m = {'name': name}
            if version is not None:
                m['version'] = version
            m.update(loader[2])
            result.append(m)

        return result
