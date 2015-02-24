#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import glob
import os
import sys

# Ensure that astropy_helpers is available. If not installed, use
# the included ah_bootstrap module to download from PyPI for temporary 
# installation. This option doesn't permanently install the package.
try:
    import astropy_helpers
except ImportError:
    import ah_bootstrap
    ah_bootstrap.use_astropy_helpers(use_git=False, auto_upgrade=False)

from setuptools import setup

#A dirty hack to get around some early import/configurations ambiguities
if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins
builtins._ASTROPY_SETUP_ = True

from astropy_helpers.setup_helpers import (
    register_commands, adjust_compiler, get_debug_option, get_package_info)
from astropy_helpers.git_helpers import get_git_devstr
from astropy_helpers.version_helpers import generate_version_py

# Need a recursive glob to find all package data files if there are
# subdirectories
import fnmatch
def recursive_glob(basedir, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(basedir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

# package-specific values
PACKAGENAME = 'sncosmo'
DESCRIPTION = 'Package for supernova cosmology based on astropy'
LONG_DESCRIPTION = ''
AUTHOR = 'The SNCosmo Developers'
AUTHOR_EMAIL = 'kylebarbary@gmail.com'
LICENSE = 'BSD'
URL = 'http://sncosmo.readthedocs.org'

# Store the package name in a built-in variable so it's easy
# to get from other parts of the setup infrastructure
builtins._ASTROPY_PACKAGE_NAME_ = PACKAGENAME

# VERSION should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
VERSION = '1.1.dev'

# Indicates if this version is a release version
RELEASE = 'dev' not in VERSION

if not RELEASE:
    VERSION += get_git_devstr(False)

# Populate the dict of setup command overrides; this should be done before
# invoking any other functionality from distutils since it can potentially
# modify distutils' behavior.
cmdclassd = register_commands(PACKAGENAME, VERSION, RELEASE)

# Adjust the compiler in case the default on this platform is to use a
# broken one.
adjust_compiler(PACKAGENAME)

# Freeze build information in version.py
generate_version_py(PACKAGENAME, VERSION, RELEASE,
                    get_debug_option(PACKAGENAME))

# The current scripts in sncosmo are for developer use, not intended to
# be installed.
scripts = []

# Get configuration information from all of the various subpackages.
# See the docstring for setup_helpers.update_package_files for more
# details.
package_info = get_package_info()

# Add the project-global data
pkgdatadir = os.path.join(PACKAGENAME, 'data')
testdatadir = os.path.join(PACKAGENAME, 'tests', 'data')

data_files = []
data_files.extend(recursive_glob(pkgdatadir, '*'))
data_files.extend(recursive_glob(testdatadir, '*'))
data_files.append(os.path.join(PACKAGENAME, 'tests', 'coveragerc'))
data_files.append(os.path.join(PACKAGENAME, PACKAGENAME + '.cfg'))
data_files = [f[len(PACKAGENAME)+1:] for f in data_files]
package_info['package_data'][PACKAGENAME] = data_files

setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      scripts=scripts,
      requires=['numpy', 'scipy', 'astropy'],
      install_requires=['numpy', 'scipy', 'astropy'],
      provides=[PACKAGENAME],
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      long_description=LONG_DESCRIPTION,
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics'],
      cmdclass=cmdclassd,
      zip_safe=False,
      use_2to3=False,
      **package_info)
