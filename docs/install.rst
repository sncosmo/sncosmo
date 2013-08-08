************
Installation
************

Requirements
============

SNCosmo depends on the following standard scientific python packages:

- `Python <http://www.python.org/>`_ 2.6 or 2.7

- `NumPy <http://www.numpy.org/>`_ 1.5 or later

- `SciPy <http://www.scipy.org/>`_ 0.9 or later

- AstroPy_ 0.2 or later

In addition, several optional packages provide additional functionality:

- Optional: `matplotlib <http://www.matplotlib.org/>`_ for plotting functions.

- Optional: `iminuit <http://iminuit.github.io/iminuit/>`_ for light curve
  fitting using the Minuit minimizer.

- Optional: `emcee <http://dan.iel.fm/emcee/>`_ for Monte Carlo parameter
  estimation.

Installation instructions
=========================

Assuming you already have NumPy and SciPy, install AstroPy by
following the `AstroPy installation instructions
<http://astropy.readthedocs.org/en/latest/install.html>`_. In short::

    pip install --no-deps astropy

Latest released version (using pip)
-----------------------------------

To install with `pip`, simply run one of::

    pip install sncosmo --no-deps
    pip install sncosmo --no-deps --user
    pip install sncosmo --no-deps --prefix=/path/to/install/dir

The first option will try to install to the system directories, and
requires root access. The ``--user`` option will typically install
things in ``~/.local/lib``, ``~/.local/bin``, etc (on Linux
systems). The ``--prefix`` option will install in
``/path/to/install/dir/lib``, ``/path/to/install/dir/bin``, etc.

Development version (using git)
-------------------------------

To get the latest development version source, using ``git``::

    git clone git://github.com/kbarbary/sncosmo.git
    cd sncosmo
    setup.py build

then one of::

    setup.py install
    setup.py install --user
    setup.py install --prefix=/path/to/prefix

Development version (no git)
----------------------------

If you don't have git but want to use the latest development version,
download the latest tarball, using ::

    wget https://github.com/kbarbary/sncosmo/archive/master.zip
    unzip master.zip
    cd sncosmo-master
    setup.py build
    setup.py install [--user] [--prefix=...]
