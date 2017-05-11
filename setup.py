#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import fnmatch
import glob
import os
import re
import sys

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.test import test as TestCommand


# Need a recursive glob to find all package data files if there are
# subdirectories. Doesn't exist on Python 2, so write our own:
def recursive_glob(basedir, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(basedir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


# class to hook up `setup.py test` to `sncosmo.test(...)`
class SNCosmoTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest"),
                    ('remote-data', None,
                     "Run tests marked with @remote_data. These tests use "
                     "online data and are not run by default."),
                    ('coverage', None,
                     "Generate a test coverage report. The result will be "
                     "placed in the directory htmlcov.")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = None
        self.remote_data = False
        self.coverage = False

    def run_tests(self):
        import shlex
        import sncosmo
        errno = sncosmo.test(args=self.pytest_args,
                             remote_data=self.remote_data,
                             coverage=self.coverage)
        sys.exit(errno)

# extension module(s): only add if setup.py argument is not egg_info, because
# we need to import numpy, and we'd rather egg_info work when dependencies
# are not installed.
if sys.argv[1] != 'egg_info':
    import numpy
    fname = os.path.join("sncosmo", "salt2utils.pyx")
    if os.path.exists(fname):
        USE_CYTHON = True
    else:
        USE_CYTHON = False
        fname = os.path.join("sncosmo", "salt2utils.c")

    source_files = [fname]
    include_dirs = [numpy.get_include()]
    extensions = [Extension("sncosmo.salt2utils", source_files,
                            include_dirs=include_dirs)]

    if USE_CYTHON:
        from Cython.Build import cythonize
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

# Synchronize version from code.
VERSION = re.findall(r"__version__ = \"(.*?)\"",
                     open(os.path.join("sncosmo", "__init__.py")).read())[0]

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
      install_requires=['numpy>=1.5.0',
                        'scipy>=0.9.0',
                        'extinction>=0.2.2',
                        'astropy>=0.4.0'],
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
      cmdclass={'test': SNCosmoTest},
      zip_safe=False,
      use_2to3=False,
      ext_modules=extensions,
      packages=['sncosmo', 'sncosmo.tests'],
      package_data={'sncosmo': data_files})
