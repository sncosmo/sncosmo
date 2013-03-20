Installation
============

Requirements
------------

SNCosmo depends on AstroPy and its dependencies. The full requirements
are:

- `Python <http://www.python.org/>`_ 2.6, 2.7, 3.1 or 3.2

- `NumPy <http://www.numpy.org/>`_ 1.4 or later

- `SciPy <http://www.scipy.org/>`_ 

- AstroPy_ 0.2 or later

Assuming you already have NumPy and SciPy, install AstroPy by
following the `AstroPy installation instructions
<http://astropy.readthedocs.org/en/v0.2/install.html>`_.

Install from source
-------------------

SNCosmo is only available from source. Either download the latest
tarball, using ::

  $ wget https://github.com/kbarbary/sncosmo/archive/master.zip
  $ unzip master.zip
  $ cd sncosmo-master
  $ setup.py build
  $ setup.py install

or clone the repository using ::

  $ git clone git://github.com/kbarbary/sncosmo.git
  $ cd sncosmo
  $ setup.py build
  $ setup.py install

If you don't have root access, install using ::

  $ setup.py install --user

Source
------

The source code is hosted on github: https://github.com/kbarbary/sncosmo .
