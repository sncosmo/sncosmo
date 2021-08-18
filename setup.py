#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy
from Cython.Build import cythonize
import os

from setuptools import setup
from setuptools.extension import Extension

# Extensions
source_files = [os.path.join("sncosmo", "salt2utils.pyx")]
include_dirs = [numpy.get_include()]
extensions = [Extension("sncosmo.salt2utils", source_files,
                        include_dirs=include_dirs)]
extensions = cythonize(extensions)

setup(use_scm_version=True, ext_modules=extensions)
