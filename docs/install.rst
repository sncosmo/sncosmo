************
Installation
************

Requirements
============

SNCosmo depends on the following standard scientific python packages:

- `Python <http://www.python.org/>`_ 2.6 or 2.7

- `NumPy <http://www.numpy.org/>`_ 1.6 or later

- `SciPy <http://www.scipy.org/>`_ 0.9 or later

- AstroPy_ 0.4 or later (`Installation instructions <http://astropy.readthedocs.org/en/stable/install.html>`_)

In addition, several packages provide optional functionality:

- Optional: `matplotlib <http://www.matplotlib.org/>`_ for plotting functions.

- Optional: `iminuit <http://iminuit.github.io/iminuit/>`_ for light curve
  fitting using the Minuit minimizer in `sncosmo.fit_lc`.

- Optional: `emcee <http://dan.iel.fm/emcee/>`_ for Monte Carlo parameter
  estimation in `sncosmo.mcmc_lc`.

The `triangle <https://github.com/dfm/triangle.py>`_ package is also
recommended for plotting results from the samplers `sncosmo.mcmc_lc`
and `sncosmo.nest_lc`, but triangle is not used by any part of
sncosmo.


Installation instructions
=========================

Use pip::

    pip install --no-deps sncosmo

.. note::

    The ``--no-deps`` flag is optional, but highly recommended if you already
    have Numpy installed, since otherwise pip will sometimes try to "help" you
    by upgrading your Numpy installation, which may not always be desired.

.. note::

    If you get a ``PermissionError`` this means that you do not have the
    required administrative access to install new packages to your Python
    installation.  In this case you may consider using the ``--user`` option
    to install the package into your home directory.  You can read more
    about how to do this in the `pip documentation
    <http://www.pip-installer.org/en/1.2.1/other-tools.html#using-pip-with-the-user-scheme>`_.

    Do **not** install Astropy or other third-party packages using ``sudo``
    unless you are fully aware of the risks.

.. note::

    You will need a C compiler (e.g. ``gcc`` or ``clang``) to be
    installed for the installation to succeed.


Development version
-------------------

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
