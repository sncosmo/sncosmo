Installation
============

Requirements
------------

SNCosmo depends on AstroPy and its dependencies. The requirements
are:

- `Python <http://www.python.org/>`_ 2.6, 2.7, 3.1 or 3.2

- `NumPy <http://www.numpy.org/>`_ 1.4 or later

- AstroPy_ 0.2 or later

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
