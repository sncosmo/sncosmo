#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import re

from setuptools import setup
from extension_helpers import get_extensions

# Synchronize version from code.
VERSION = re.findall(r"__version__ = \"(.*?)\"",
                     open(os.path.join("sncosmo", "__init__.py")).read())[0]

setup(
    version=VERSION,
    ext_modules=get_extensions()
)
