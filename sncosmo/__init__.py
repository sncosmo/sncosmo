# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sncosmo: A Python package for supernova cosmology
"""

from __future__ import absolute_import

import os
from astropy.config import ConfigItem, ConfigNamespace
from astropy.config.configuration import update_default_config


__version__ = "1.6.dev0"

def test(package=None, test_path=None, args=None, plugins=None,
         verbose=False, pastebin=None, remote_data=False, pep8=False,
         pdb=False, coverage=False, open_files=False, **kwargs):
    """
    Run the tests using py.test. A proper set of arguments is constructed and
    passed to `pytest.main`.

    Parameters
    ----------
    package : str, optional
        The name of a specific package to test, e.g. 'io.fits' or 'utils'.
        If nothing is specified all default tests are run.

    test_path : str, optional
        Specify location to test by path. May be a single file or
        directory. Must be specified absolutely or relative to the
        calling directory.

    args : str, optional
        Additional arguments to be passed to `pytest.main` in the `args`
        keyword argument.

    plugins : list, optional
        Plugins to be passed to `pytest.main` in the `plugins` keyword
        argument.

    verbose : bool, optional
        Convenience option to turn on verbose output from py.test. Passing
        True is the same as specifying `-v` in `args`.

    pastebin : {'failed','all',None}, optional
        Convenience option for turning on py.test pastebin output. Set to
        'failed' to upload info for failed tests, or 'all' to upload info
        for all tests.

    remote_data : bool, optional
        Controls whether to run tests marked with @remote_data. These
        tests use online data and are not run by default. Set to True to
        run these tests.

    pep8 : bool, optional
        Turn on PEP8 checking via the pytest-pep8 plugin and disable normal
        tests. Same as specifying `--pep8 -k pep8` in `args`.

    pdb : bool, optional
        Turn on PDB post-mortem analysis for failing tests. Same as
        specifying `--pdb` in `args`.

    coverage : bool, optional
        Generate a test coverage report.  The result will be placed in
        the directory htmlcov.

    open_files : bool, optional
        Fail when any tests leave files open.  Off by default, because
        this adds extra run time to the test suite.  Works only on
        platforms with a working ``lsof`` command.

    parallel : int, optional
        When provided, run the tests in parallel on the specified
        number of CPUs.  If parallel is negative, it will use the all
        the cores on the machine.  Requires the
        `pytest-xdist <https://pypi.python.org/pypi/pytest-xdist>`_ plugin
        installed. Only available when using Astropy 0.3 or later.

    kwargs
        Any additional keywords passed into this function will be passed
        on to the astropy test runner.  This allows use of test-related
        functionality implemented in later versions of astropy without
        explicitly updating the package template.

    See Also
    --------
    pytest.main : py.test function wrapped by `run_tests`.

    """
    import os
    from astropy.tests.helper import TestRunner

    runner = TestRunner(os.path.dirname(__file__))

    return runner.run_tests(
        package=package, test_path=test_path, args=args,
        plugins=plugins, verbose=verbose, pastebin=pastebin,
        remote_data=remote_data, pep8=pep8, pdb=pdb,
        coverage=coverage, open_files=open_files, **kwargs)


# Create default configurations. The file sncosmo.cfg should be
# kept in sync with the ConfigItems here.
class _Conf(ConfigNamespace):
    """Configuration parameters for sncosmo."""
    data_dir = ConfigItem(
        None,
        "Directory where sncosmo will store and read downloaded data "
        "resources. If None, ASTROPY_CACHE_DIR/sncosmo is created and "
        "used. Example: data_dir = /home/user/data/sncosmo",
        cfgtype='string(default=None)')
    sfd98_dir = ConfigItem(
        None,
        "Directory containing SFD (1998) dust maps, with names: "
        "'SFD_dust_4096_ngp.fits' and 'SFD_dust_4096_sgp.fits'. "
        "Example: sfd98_dir = /home/user/data/sfd98",
        cfgtype='string(default=None)')
    remote_timeout = ConfigItem(
        10.0, "Remote timeout in seconds.")

# Create an instance of the class we just defined.
conf = _Conf()

# Update the user's ~/.astropy/config/sncosmo.cfg if needed.
update_default_config("sncosmo",  # pkg
                      os.path.dirname(__file__),  # configdir
                      version=__version__)

# clean up namespace
del os, ConfigItem, ConfigNamespace, update_default_config

# import all the things into the top-level namespace
from .bandpasses import *
from .magsystems import *
from .spectrum import *
from .models import *
from .io import *
from .snanaio import *
from .fitting import *
from .simulation import *
from .plotting import *
from .photdata import *
from .registry import *

# deprecated stuff
from . import registry  # deprecated in v1.2; use previous import.
from ._deprecated import *

# Register all the built-ins.
from .builtins import *
