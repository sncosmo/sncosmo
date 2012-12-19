Installation
============

Requirements:

* Python 2.6 or greater
* NumPy 1.5 or greater
* AstroPy 0.2 or greater

Install NumPy and AstroPy following the instructions on the relevant sites. To install :mod:`SNSim`, download the latest tarball, using ::

  $ wget https://github.com/kbarbary/snsim/archive/master.zip
  $ unzip master.zip
  $ cd snsim-master
  $ setup.py build
  $ setup.py install

or ::

  $ git clone git://github.com/kbarbary/snsim.git
  $ cd snsim
  $ setup.py build
  $ setup.py install

If you don't have root access, install using ::

  $ setup.py install --user

Getting Package Data
--------------------

:mod:`SNSim` includes "factory functions" for generating ``Spectrum``, ``Bandpass`` and ``TransientModel`` instances from built-in data. For these to work, the data must be downloaded ::

  $ wget https://www.dropbox.com/s/m9irlhe7oq6a3ho/snsim-data.tar.gz
  $ tar xvzf snsim-data.tar.gz

When the factory functions are called, the relevant data is located using the path given in the user's ``SNSIM_DATA`` environment variable. In bash, ::

  $ export SNSIM_DATA=/path/to/snsim-data

