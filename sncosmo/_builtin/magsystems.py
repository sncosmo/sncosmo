# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Importing this module registers loaders for built-in magnitude systems.
# The module docstring, a table of the magnitude systems, is generated at the
# after all the bandpasses are registered.

from .. import registry
from .. import MagnitudeSystem, ABSystem


def load_ab():
    return ABSystem()

registry.register_loader(MagnitudeSystem, 'ab', load_ab)
