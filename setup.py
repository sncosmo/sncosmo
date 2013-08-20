#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst


import sys
import imp
try:
    # This incantation forces distribute to be used (over setuptools) if it is
    # available on the path; otherwise distribute will be downloaded.
    import pkg_resources
    distribute = pkg_resources.get_distribution('distribute')
    if pkg_resources.get_distribution('setuptools') != distribute:
        sys.path.insert(1, distribute.location)
        distribute.activate()
        imp.reload(pkg_resources)
except:  # There are several types of exceptions that can occur here
    from distribute_setup import use_setuptools
    use_setuptools()

import glob
import os
from setuptools import setup, find_packages


# Need a recursive glob to find all package data files if there are
# subdirectories
import fnmatch
def recursive_glob(basedir, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(basedir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

#A dirty hack to get around some early import/configurations ambiguities
#This is the same as setup_helpers.set_build_mode(), but does not require
#importing setup_helpers
if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins
builtins._PACKAGE_SETUP_ = True

import astropy
from astropy.setup_helpers import (register_commands, adjust_compiler,
                                   filter_packages, update_package_files,
                                   get_debug_option)
from astropy.version_helpers import get_git_devstr, generate_version_py

# Set affiliated package-specific settings
PACKAGENAME = 'sncosmo'
DESCRIPTION = 'Package for supernova cosmology based on astropy'
LONG_DESCRIPTION = ''
AUTHOR = 'The SNCosmo Developers'
AUTHOR_EMAIL = 'kylebarbary@gmail.com'
LICENSE = 'BSD'
URL = 'http://sncosmo.readthedocs.org'

#VERSION should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
VERSION = '0.3.dev'

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

# Use the find_packages tool to locate all packages and modules
packagenames = filter_packages(find_packages())

# Treat everything in scripts except README.rst as a script to be installed
ignore_scripts = ['generate_example_data.py']
scripts = [fname for fname in glob.glob(os.path.join('scripts', '*'))
           if (os.path.basename(fname) != 'README.rst' and
               fname[-1] != '~' and
               os.path.basename(fname) not in ignore_scripts)]

# This dictionary stores the command classes used in setup below
#cmdclassd = {'test': setup_helpers.setup_test_command(PACKAGENAME),

             # Use distutils' sdist because it respects package_data.
             # setuptools/distributes sdist requires duplication of
             # information in MANIFEST.in
#             'sdist': sdist.sdist,

             # Use a custom build command which understands additional
             # commandline arguments
#             'build': setup_helpers.AstropyBuild,

             # Use a custom install command which understands additional
             # commandline arguments
#             'install': setup_helpers.AstropyInstall

#             }

#if setup_helpers.HAVE_CYTHON and not RELEASE:
#    from Cython.Distutils import build_ext
    # Builds Cython->C if in dev mode and Cython is present
#    cmdclassd['build_ext'] = setup_helpers.wrap_build_ext(build_ext)
#else:
#    cmdclassd['build_ext'] = setup_helpers.wrap_build_ext()

#if setup_helpers.AstropyBuildSphinx is not None:
#    cmdclassd['build_sphinx'] = setup_helpers.AstropyBuildSphinx

# Set our custom command class mapping in setup_helpers, so that
# setup_helpers.get_distutils_option will use the custom classes.
#setup_helpers.cmdclassd = cmdclassd

# Additional C extensions that are not Cython-based should be added here.
extensions = []

# A dictionary to keep track of all package data to install
data_files = recursive_glob(os.path.join(PACKAGENAME, 'data'), '*')
data_files = [f[len(PACKAGENAME)+1:] for f in data_files]
package_data = {PACKAGENAME: data_files}

# A dictionary to keep track of extra packagedir mappings
package_dirs = {}

skip_2to3 = []

# Update extensions, package_data, packagenames and package_dirs from
# any sub-packages that define their own extension modules and package
# data.  See the docstring for setup_helpers.update_package_files for
# more details.
update_package_files(PACKAGENAME, extensions, package_data,
                     packagenames, package_dirs)

setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      packages=packagenames,
      package_data=package_data,
      package_dir=package_dirs,
      ext_modules=extensions,
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
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      cmdclass=cmdclassd,
      zip_safe=False,
      use_2to3=True
      )
