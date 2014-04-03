#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import glob
import os
import sys

import setuptools_bootstrap
from setuptools import setup

#A dirty hack to get around some early import/configurations ambiguities
if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins
builtins._ASTROPY_SETUP_ = True

import astropy
from astropy.setup_helpers import (register_commands, adjust_compiler,
                                   get_package_info, get_debug_option,
                                   is_distutils_display_option)
from astropy.version_helpers import get_git_devstr, generate_version_py

# Need a recursive glob to find all package data files if there are
# subdirectories
import fnmatch
def recursive_glob(basedir, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(basedir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

# Set affiliated package-specific settings
PACKAGENAME = 'sncosmo'
DESCRIPTION = 'Package for supernova cosmology based on astropy'
LONG_DESCRIPTION = ''
AUTHOR = 'The SNCosmo Developers'
AUTHOR_EMAIL = 'kylebarbary@gmail.com'
LICENSE = 'BSD'
URL = 'http://sncosmo.readthedocs.org'

#VERSION should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
VERSION = '0.4.1'

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
generate_version_py(PACKAGENAME, VERSION, RELEASE, get_debug_option())

# Treat everything in scripts except README.rst as a script to be installed
#ignore_scripts = ['generate_example_data.py']
#scripts = [fname for fname in glob.glob(os.path.join('scripts', '*'))
#           if (os.path.basename(fname) != 'README.rst' and
#               fname[-1] != '~' and
#               os.path.basename(fname) not in ignore_scripts)]
scripts = []

# Get configuration information from all of the various subpackages.
# See the docstring for setup_helpers.update_package_files for more
# details.
package_info = get_package_info(PACKAGENAME)

# Add the project-global data
data_files = recursive_glob(os.path.join(PACKAGENAME, 'data'), '*')
data_files = [f[len(PACKAGENAME)+1:] for f in data_files]
package_info['package_data'][PACKAGENAME] = data_files

install_requires = ['numpy', 'scipy', 'astropy >= 0.3']
setup_requires = []

# Avoid installing setup_requires dependencies if the user just
# queries for information
if is_distutils_display_option():
    setup_requires = []

setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      scripts=scripts,
      requires=['numpy', 'scipy', 'astropy'],
      install_requires=install_requires,
      setup_requires=setup_requires,
      provides=[PACKAGENAME],
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      long_description=LONG_DESCRIPTION,
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      cmdclass=cmdclassd,
      zip_safe=False,
      use_2to3=True,
      **package_info
      )
