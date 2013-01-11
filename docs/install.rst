Installation
============

Requirements
------------

* Python 2.6 or greater
* NumPy 1.5 or greater
* AstroPy 0.2 or greater

Install from source
-------------------
:mod:`SNSim` is only available from source. Either download the latest tarball, using ::

  $ wget https://github.com/kbarbary/snsim/archive/master.zip
  $ unzip master.zip
  $ cd snsim-master
  $ setup.py build
  $ setup.py install

or clone the repository using ::

  $ git clone git://github.com/kbarbary/snsim.git
  $ cd snsim
  $ setup.py build
  $ setup.py install

If you don't have root access, install using ::

  $ setup.py install --user

Or, if you want to contribute, fork the `GitHub repostory <https://github.com/kbarbary/snsim>`_!

Package Data
------------

:mod:`SNSim` includes "factory functions" for generating ``Spectrum``, ``Bandpass`` and ``TransientModel`` instances from built-in data. For these to work, the data must be downloaded ::

  $ wget www.hep.anl.gov/kbarbary/project-data/snsim/snsim-data-v0.1.dev.tar.gz
  $ tar xvzf snsim-data-v0.1.dev.tar.gz

When the factory functions are called, the relevant data is located using the path given in the user's ``SNSIM_DATA`` environment variable. In bash, ::

  $ export SNSIM_DATA=/path/to/snsim-data

