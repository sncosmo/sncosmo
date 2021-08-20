#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy
import os
import re

from setuptools import setup
from setuptools.extension import Extension

# Synchronize version from code.
VERSION = re.findall(r"__version__ = \"(.*?)\"",
                     open(os.path.join("sncosmo", "__init__.py")).read())[0]

# Cython extensions
extensions = [
    Extension(
        "sncosmo.salt2utils",
        sources=[os.path.join("sncosmo", "salt2utils.pyx")],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    version=VERSION,
    ext_modules=extensions,
)
