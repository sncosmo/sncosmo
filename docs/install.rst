************
Installation
************

SNCosmo works on Python 2.6, 2.7 and Python 3.3+ and requires the
following standard scientific python packages: `NumPy
<http://www.numpy.org/>`_, `SciPy <http://www.scipy.org/>`_ and
AstroPy_.


Install using conda
===================

SNCosmo is available on the `astropy channel <https://anaconda.org/astropy/>`_::

    conda install -c astropy sncosmo

The release of the conda version may lag behind the pip release, so
you may want to check the available conda version. You can do this
at the link above, or by running ``conda search -c astropy sncosmo``.


Install using pip
=================

Using pip::

    pip install --no-deps sncosmo

.. note::

    The ``--no-deps`` flag is optional, but highly recommended if you
    already have Numpy installed, since otherwise pip will sometimes
    try to "help" you by upgrading your Numpy installation, which may
    not always be desired.

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

    ./setup.py install

As with the pip install instructions, you may want to use either
``setup.py install --user`` or ``setup.py develop`` to alter where the
package is installed.


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

iminuit, emcee and nestle can be installed using pip.

The `corner <https://github.com/dfm/corner.py>`_ package is also
recommended for plotting results from the samplers `sncosmo.mcmc_lc`
and `sncosmo.nest_lc`, but is not used by any part of sncosmo.
