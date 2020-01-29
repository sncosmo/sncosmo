#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import fnmatch
import glob
import os
import re
import sys

from setuptools import setup
from setuptools.extension import Extension


# Need a recursive glob to find all package data files if there are
# subdirectories. Doesn't exist on Python 3.4, so write our own:
def recursive_glob(basedir, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(basedir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


# Synchronize version from code.
VERSION = re.findall(r"__version__ = \"(.*?)\"",
                     open(os.path.join("sncosmo", "__init__.py")).read())[0]


# extension module(s): only add if setup.py argument is not egg_info, because
# we need to import numpy, and we'd rather egg_info work when dependencies
# are not installed.
if sys.argv[1] != 'egg_info':
    import numpy
    from Cython.Build import cythonize
    source_files = [os.path.join("sncosmo", "salt2utils.pyx")]
    include_dirs = [numpy.get_include()]
    extensions = [Extension("sncosmo.salt2utils", source_files,
                            include_dirs=include_dirs)]
    extensions = cythonize(extensions)
else:
    extensions = None

PACKAGENAME = 'sncosmo'
DESCRIPTION = 'Package for supernova cosmology based on astropy'
LONG_DESCRIPTION = ''
AUTHOR = 'The SNCosmo Developers'
AUTHOR_EMAIL = 'kylebarbary@gmail.com'
LICENSE = 'BSD'
URL = 'http://sncosmo.readthedocs.org'

# Add the project-global data
pkgdatadir = os.path.join(PACKAGENAME, 'data')
testdatadir = os.path.join(PACKAGENAME, 'tests', 'data')
data_files = []
data_files.extend(recursive_glob(pkgdatadir, '*'))
data_files.extend(recursive_glob(testdatadir, '*'))
data_files.append(os.path.join(PACKAGENAME, 'tests', 'coveragerc'))
data_files.append(os.path.join(PACKAGENAME, PACKAGENAME + '.cfg'))
data_files = [f[len(PACKAGENAME)+1:] for f in data_files]

setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      python_requires='>=3.5',
      install_requires=['numpy>=1.13.3',
                        'scipy>=0.9.0',
                        'extinction>=0.2.2',
                        'astropy>=1.0.0'],
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
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics'],

      zip_safe=False,
      use_2to3=False,
      ext_modules=extensions,
      packages=['sncosmo', 'sncosmo.tests'],
      package_data={'sncosmo': data_files})
