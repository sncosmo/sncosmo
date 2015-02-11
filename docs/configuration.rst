***********************
Directory Configuration
***********************

The "built-in" Sources and Spectra in SNCosmo depend on some sizable
data files. These files are hosted remotely, downloaded as needed, and
cached locally. This all happens automatically, but it is helpful to
know where the files are stored if you want to inspect them or share a
common download directory between multiple users.

By default, SNCosmo will create and use an ``sncosmo`` subdirectory in
the AstroPy cache directory for this purpose. For example,
``$HOME/.astropy/cache/sncosmo``. After using a few models and spectra
for the first time, here is what that directory might look like::

    $ tree ~/.astropy/cache/sncosmo
    /home/kyle/.astropy/cache/sncosmo
    ├── models
    │   ├── hsiao
    │   │   └── Hsiao_SED_V3.fits
    │   └── sako
    │       ├── S11_SDSS-000018.SED
    │       ├── S11_SDSS-001472.SED
    │       └── S11_SDSS-002000.SED
    └── spectra
        ├── alpha_lyr_stis_007.fits
        └── bd_17d4708_stisnic_005.fits

You can see that within the top-level ``$HOME/.astropy/cache/sncosmo``
directory, a particular directory structure is created. This directory
structure is fixed in the code, so it's best not to move things around
within the top-level directory. If you do, sncosmo will think the data
have not been downloaded and will re-download them.


Configuring the Directories
===========================

What if you would rather use a different directory to store downloaded
data?  Perhaps you'd rather the data not be in a hidden directory, or
perhaps there are multiple users who wish to use a shared data
directory. This is where the configuration file comes in. This file is
found in the astropy configuration directory, e.g.,
``$HOME/.astropy/config/sncosmo.cfg``. When you ``import sncosmo`` it
checks for this file and creates a default one if it doesn't
exist. The default one looks like this::

    $ cat ~/.astropy/config/sncosmo.cfg 

    ## Directory containing SFD (1998) dust maps, with names:
    ## 'SFD_dust_4096_ngp.fits' and 'SFD_dust_4096_sgp.fits'
    ## Example: sfd98_dir = /home/user/data/sfd98
    # sfd98_dir = None

    ## Directory where sncosmo will store and read downloaded data resources.
    ## If None, ASTROPY_CACHE_DIR/sncosmo will be used.
    ## Example: data_dir = /home/user/data/sncosmo
    # data_dir = None

To change the data directory, simply uncomment the last line and set it to the
desired directory. You can even move the data directory around, as long as you
update this configuration parameter accordingly.
