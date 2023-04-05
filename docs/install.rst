************
Installation
************

SNCosmo works on Python 3.7+ and depends on the
following Python packages:

- `numpy <http://www.numpy.org/>`_
- `scipy <http://www.scipy.org/>`_
- `astropy <http://www.astropy.org>`_
- `extinction <http://extinction.readthedocs.io>`_


Install using conda (recommended)
=================================

If you are using Anaconda or the conda package manager, you can install SNCosmo
from the conda-forge channel::

    conda install -c conda-forge sncosmo


Install using pip
=================

First ensure that numpy and cython are installed. Then::

    pip install sncosmo

.. note::

    The ``--no-deps`` flag is optional, but highly recommended if you
    already have numpy, scipy and astropy installed, since otherwise
    pip will sometimes try to "help" you by upgrading your Numpy
    installation, which may not always be desired.

.. note::

    If you get a ``PermissionError`` this means that you do not have
    the required administrative access to install new packages to your
    Python installation.  In this case you may consider using the
    ``--user`` option to install the package into your home directory.
    You can read more about how to do this in the `pip documentation
    <https://pip.pypa.io/en/latest/user_guide.html#user-installs>`_.

    Do **not** install sncosmo or other third-party packages using
    ``sudo`` unless you are fully aware of the risks.

.. note::

    You will need a C compiler (e.g. ``gcc`` or ``clang``) to be
    installed for the installation to succeed.


Install latest development version
==================================

SNCosmo is being developed `on github
<https://github.com/sncosmo/sncosmo>`_. To get the latest development
version using ``git``::

    git clone git://github.com/sncosmo/sncosmo.git
    cd sncosmo

then::

    pip install -e .

This will install a development version of the SNCosmo package that
automatically picks up any changes that you made when you import sncosmo for
the first time in a Python interpreter. If you make any edits to the Cython
code in SNCosmo (files with .c or .pyx extensions), then you will need to run
this command again to compile that code for your changes to be picked up.


Optional dependencies
=====================

Several additional packages are recommended for enabling optional
functionality in SNCosmo.

- `matplotlib <http://www.matplotlib.org/>`_ for plotting
  functions.
- `iminuit <http://iminuit.github.io/iminuit/>`_ for light curve
  fitting using the Minuit minimizer in `sncosmo.fit_lc`.
- `emcee <http://dan.iel.fm/emcee/>`_ for MCMC light curve parameter
  estimation in `sncosmo.mcmc_lc`.
- `nestle <http://kbarbary.github.io/nestle/>`_ for nested sampling
  light curve parameter estimation in `sncosmo.nest_lc`.

The `corner <https://github.com/dfm/corner.py>`_ package is also
recommended for plotting results from the samplers `sncosmo.mcmc_lc`
and `sncosmo.nest_lc`, but is not used by any part of sncosmo.
