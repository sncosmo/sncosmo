Installation
============

Requirements
------------

SNCosmo depends on the following standard scientific python packages:

- `Python <http://www.python.org/>`_ 2.6, 2.7

- `NumPy <http://www.numpy.org/>`_ 1.4 or later

- `SciPy <http://www.scipy.org/>`_ (tested on 0.10.1)

- AstroPy_ 0.2 or later

- Optional: `matplotlib <http://www.matplotlib.org/>`_ for plotting functions.

Assuming you already have NumPy and SciPy, install AstroPy by
following the `AstroPy installation instructions
<http://astropy.readthedocs.org/en/latest/install.html>`_.

Installation instructions
-------------------------

Either download the latest tarball, using ::

    wget https://github.com/kbarbary/sncosmo/archive/master.zip
    unzip master.zip
    cd sncosmo-master
    setup.py build
    setup.py install

or clone the repository using ::

    git clone git://github.com/kbarbary/sncosmo.git
    cd sncosmo
    setup.py build
    setup.py install

If you don't have root access, install using ::

    setup.py install --user

or ::

    setup.py install --prefix=/path/to/prefix


Source
------

The source code is hosted on github: https://github.com/kbarbary/sncosmo
