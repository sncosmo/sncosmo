Installation
============

Requirements
------------

* Python 2.6 or greater
* NumPy 1.5 or greater
* AstroPy 0.2 or greater

Install from source
-------------------

:mod:`SNSim` is only available from source. Either download the latest
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

Or, if you want to contribute, fork the `GitHub repostory <https://github.com/kbarbary/sncosmo>`_!

Package Data
------------

:mod:`SNSim` includes "factory functions" for generating ``Spectrum``, ``Bandpass`` and ``TransientModel`` instances from built-in data. For these to work, the data must be downloaded ::

  $ wget kbarbary.github.com/data/sncosmo-data-v0.1.dev.tar.gz
  $ tar xvzf sncosmo-data-v0.1.dev.tar.gz

When the factory functions are called, the relevant data is located using the path given in the user's ``SNCOSMO_DATA`` environment variable. In bash, ::

  $ export SNCOSMO_DATA=/path/to/sncosmo-data

