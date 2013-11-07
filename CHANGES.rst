0.3.0 (2013-11-07)
----------------

This is a release with mostly bugfixes but a few new features, designed to be
backwards compatible with v0.2.0 ahead of API changes coming in the next
version.

- New functions:

  - ``sncosmo.get_ebv_from_map``: E(B-V) at given coordinates from SFD map. 
  - ``sncosmo.read_snana_ascii``: Read SNANA ascii format files.
  - ``sncosmo.read_snana_fits``: Read SNANA FITS format files.
  - ``sncosmo.read_snana_simlib``: Read SNANA ascii "SIMLIB" files.

0.2 (2013-08-20)
----------------

- New built-ins

  - Added SN 2011fe Nearby Supernova Factory data to built-in models as
    ``'2011fe'``

- Renamed functions (some with enhancements), previously "experimental":
  - ``sncosmo.fit_lc`` (previously ``sncosmo.fit_model``)
  - ``sncosmo.read_lc`` (previously ``sncosmo.readlc``)
  - ``sncosmo.write_lc`` (previously ``sncosmo.writelc``)
  - ``sncosmo.plot_lc`` (previously ``sncosmo.plotlc``)

- New functions:

  - ``sncosmo.load_example_data``: Example photometric data.
  - ``sncosmo.mcmc_lc``: Markov Chain Monte Carlo parameter estimation.
  - ``sncosmo.animate_model``: Model animation using matplotlib.animation.

- ``sncosmo.fit_lc``: Now uses the iminuit package for minimization by
  default. This requires the iminuit package to be installed, but the
  old minimizer (from scipy) can still be used by setting the keyword
  ``method='l-bfgs-b'``.

- ``sncosmo.plot_lc``: Ability to plot model synthetic photometry
  without observed data, using the syntax::

      >>> sncosmo.plot_lc(model=model, bands=['band1', 'band2'])

- Photometric data format is now more flexible, allowing various names
  for table columns.


0.1 (2013-07-15)
----------------

- Initial release.
