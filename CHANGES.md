1.0.0 (unreleased)
------------------

**Enhancements:**

- Many new built-in models.
- Many new built-in bandpasses.
- New remote data fetching system. [#73]
- SALT2 model covariance available via `Model.bandfluxcov()` method and
  `modelcov=True` keyword argument passed to `fit_lc`.
- New simulation function, `zdist`, generates a distribution of redshifts
  given a volumetric rate function and cosmology.
- New simulation function, `realize_lcs`, simulates light curve data given a
  model, parameters, and observations.
- Add color-related keyword arguments to `plot_lc()`.
- Add `tighten_ylim` keyword argument to `plot_lc()`.
- Add `chisq()` function and use internally in `fit_lc()`.
- Add `SFD98Map` class for dealing with SFD (1998) dust maps persistently so
  that the underlying FITS files are opened only once. 
- Update `get_ebv_from_map()` to work with new SkyCoord class in
  `astropy.coordinates` available in astropy v0.3 onward. Previously, this
  function did not work with astropy v0.4.x (where older coordinates classes
  had been removed).
- Update to new configuration system available in astropy v0.4 onward.
  This makes this release incompatible with astropy versions less than
  0.4.
- Now compatible with Python 3.
- Increased test coverage.
- Numerous minor bugfixes.

**API changes:**

- The API of `mcmc_lc` has changed significantly (the function was marked
  experimental in previous release).

- [DEPRECATION] In result of `fit_lc`, `res.cov_names` changed to
  `res.vparam_names`.  In result of `nest_lc`, `res.param_names` and
  `res.param_dict` deprecated. This is for compatibility between
  results of `fit_lc` and `nest_lc`. [#30]

- [DEPRECATION] Deprecate `flatten` keyword argument in `fit_lc()` in
  favor of explicit use of `flatten_result()` function.


0.4.2 (2014-04-08)
------------------

This is a minor bugfix release.

- Update CALSPEC FTP base URL to the "permanent" one rather than "current".
  This should provide a stable URL, so that spectra downloads do not fail
  (even if they will not necessarily be the most current).
- Bump Vega and BD+17 spectra versions to latest CALSPEC.
- Fix an issue with astropy.utils.data.download_file() that was preventing
  the astropy.utils.data.REMOTE_DATA configuration item from having an
  effect.

0.4.1 (2014-04-03)
------------------

This is a minor bugfix release.

- Fix bug that caused `fit_lc()` and `plot_lc()` to fail on numpy 1.8.
- Correct behavior of `Model.minwave()` and `Model.maxwave()` to
  include propagation effects.
- Fix setup issue that caused cython to run even when C files already
  present.

0.4.0 (2014-03-26)
------------------

This is a non-backwards-compatible release, due to changes in the way
models are defined. These changes were made after feedback on the initial
design.

The most major change is a new central class `Model` used throughout
the pacakge. A `Model` instance encompasses a `Source` and zero or
more `PropagationEffect` instances. This is so that different
source models (e.g., SALT2 or spectral time series models) can be
combined with arbitrary dust models. The best way to think about this
is `Source` and `PropagationEffect` define the rest-frame behavior
of a SN and dust, and a `Model` puts these together to determine the
observer-frame behavior.

- New classes

  - `sncosmo.Model`: new main container class
  - `sncosmo.Source`: replaces existing `Model`
  - `sncosmo.TimeSeriesSource`: replaces existing `TimeSeriesModel`
  - `sncosmo.StretchSource`: replaces existing `StretchModel`
  - `sncosmo.SALT2Source`: replaces existing `SALT2Model`
  - `sncosmo.PropagationEffect`
  - `sncosmo.CCM89Dust`
  - `sncosmo.OD94Dust`
  - `sncosmo.F99Dust`

- New public functions

  - `sncosmo.read_griddata_ascii`: Read file with `phase wave flux` rows
  - `sncosmo.read_griddata_fits`
  - `sncosmo.write_griddata_fits`
  - `sncosmo.nest_lc`: Nested sampling parameter estimation of SN model
  - `sncosmo.simulate_vol` (EXPERIMENTAL): simulation convenience function.

- Built-ins

  - updated SALT2 model URLs
  - added SALT2 version 2.4 (Betoule et al 2014)

- Improvements to `sncosmo.plot_lc`: flexibility and layout

- Many bugfixes

0.3.0 (2013-11-07)
------------------

This is a release with mostly bugfixes but a few new features, designed to be
backwards compatible with v0.2.0 ahead of API changes coming in the next
version.

- New functions:

  - `sncosmo.get_ebv_from_map`: E(B-V) at given coordinates from SFD map. 
  - `sncosmo.read_snana_ascii`: Read SNANA ascii format files.
  - `sncosmo.read_snana_fits`: Read SNANA FITS format files.
  - `sncosmo.read_snana_simlib`: Read SNANA ascii "SIMLIB" files.

0.2 (2013-08-20)
----------------

- New built-ins

  - Added SN 2011fe Nearby Supernova Factory data to built-in models as
    `'2011fe'`

- Renamed functions (some with enhancements), previously "experimental":
  - `sncosmo.fit_lc` (previously `sncosmo.fit_model`)
  - `sncosmo.read_lc` (previously `sncosmo.readlc`)
  - `sncosmo.write_lc` (previously `sncosmo.writelc`)
  - `sncosmo.plot_lc` (previously `sncosmo.plotlc`)

- New functions:

  - `sncosmo.load_example_data`: Example photometric data.
  - `sncosmo.mcmc_lc`: Markov Chain Monte Carlo parameter estimation.
  - `sncosmo.animate_model`: Model animation using matplotlib.animation.

- `sncosmo.fit_lc`: Now uses the iminuit package for minimization by
  default. This requires the iminuit package to be installed, but the
  old minimizer (from scipy) can still be used by setting the keyword
  `method='l-bfgs-b'`.

- `sncosmo.plot_lc`: Ability to plot model synthetic photometry
  without observed data, using the syntax::
  ```
  >>> sncosmo.plot_lc(model=model, bands=['band1', 'band2'])
  ```

- Photometric data format is now more flexible, allowing various names
  for table columns.


0.1 (2013-07-15)
----------------

- Initial release.
