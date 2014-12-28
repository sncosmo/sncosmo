************
Installation
************

Requirements
============

SNCosmo depends on the following standard scientific python packages:

- `Python <http://www.python.org/>`_ 2.6 or 2.7

- `NumPy <http://www.numpy.org/>`_ 1.6 or later

- `SciPy <http://www.scipy.org/>`_ 0.9 or later

- AstroPy_ 0.4 or later

In addition, several optional packages provide additional functionality:

- Optional: `matplotlib <http://www.matplotlib.org/>`_ for plotting functions.

- Optional: `iminuit <http://iminuit.github.io/iminuit/>`_ for light curve
  fitting using the Minuit minimizer in `sncosmo.fit_lc`.

- Optional: `emcee <http://dan.iel.fm/emcee/>`_ for Monte Carlo parameter
  estimation in `sncosmo.mcmc_lc`.

- Optional: `triangle <https://github.com/dfm/triangle.py>`_ for plotting
  results from `sncosmo.mcmc_lc` and `sncosmo.nest_lc` 

Install AstroPy
---------------

Assuming you already have NumPy and SciPy, install AstroPy by
following the `AstroPy installation instructions
<http://astropy.readthedocs.org/en/latest/install.html>`_.

Installation instructions
=========================

Latest released version (using pip)
-----------------------------------

To install with `pip`, simply run one of::

    pip install sncosmo --no-deps --user
    pip install sncosmo --no-deps --install-option="--prefix=/path/to/install/basedir"
    pip install sncosmo --no-deps --target=/path/to/library/dir
    pip install sncosmo --no-deps  # as root (not recommended)

* ``--user`` will typically install things in ``~/.local/lib``,
  ``~/.local/bin``, etc (on Linux systems anyway).
* ``--install-option=...`` will install in ``/path/to/install/basedir/lib``,
  ``/path/to/install/basedir/bin``, etc.
* ``--target=...`` will install *just* the python library (no scripts)
  in ``/path/to/library/dir`` (e.g., you have ``/path/to/library/dir`` in
  your ``$PYTHONPATH``). (There are currently no scripts.)
* The last option will try to install to the system directories, and
  requires root access. I don't recommend this option unless this is
  already your typical way of using pip. I prefer to allow only the package
  manager to install to system directories, using pip only to install
  packages somewhere in my home directory using one of the above
  methods.

Latest released version (from source)
-------------------------------------

A source tarball or zip is available from `github <https://github.com/sncosmo/sncosmo/releases>`_. After downloading the appropriate tarball, run, e.g.::

    tar xzf sncosmo-0.2.tar.gz
    cd sncosmo-0.2

then one of::

    setup.py install --user
    setup.py install --prefix=/path/to/install/basedir
    setup.py install  # as root (not recommended)
    

Development version (using git)
-------------------------------

To get the latest development version source, using ``git``::

    git clone git://github.com/sncosmo/sncosmo.git
    cd sncosmo

then one of::

    setup.py install --user
    setup.py install --prefix=/path/to/install/basedir
    setup.py install  # as root (not recommended)

Development version (no git)
----------------------------

If you don't have git but want to use the latest development version,
download the latest zip, using::

    wget https://github.com/sncosmo/sncosmo/archive/master.zip
    unzip master.zip
    cd sncosmo-master
    setup.py install [--user] [--prefix=...]
