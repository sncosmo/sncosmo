==========
What's New
==========

*Note:* SNCosmo uses `Semantic Versioning <http://semver.org>`_ for its version
numbers.

What's new in v1.2.0 (2015-12-01)
=================================

API Changes
-----------

- Registry functions moved to the top-level namespace:

  - ``sncosmo.registry.register()`` -> ``sncosmo.register()``
  - ``sncosmo.registry.register_loader()`` -> ``sncosmo.register_loader()``
  - ``sncosmo.registry.retrieve()`` -> deprecated, use class-specific functions such as ``sncosmo.get_bandpass()``.

  The old import paths will still work, so this is backwards compatible.

Enhancements
------------

- ``nest_lc()`` now uses the ``nestle`` module under the hood. A new
  keyword ``method`` is available which selects different sampling
  methods implemented by ``nestle``. The new methods provide potential
  efficiency gains.
- The MLCS2k2 model is now available as a built-in Source, with the
  name ``'mlcs2k2'``.
- Bandpasses from the Carnegie Supernova Project added to built-ins.
- In ``realize_lcs()``, a new ``scatter`` keyword makes adding noise
  optional.

In addition, there have been several minor bug fixes and
documentation improvements.


What's new in v1.1.1 (2015-10-28)
=================================

This is a bugfix release for the following issue:

- Fix built-in Bessell bandpass definitions, which were wrong by a term
  proportional to inverse wavelength. This was due to misinterpretation
  of the trasmission units.


What's new in v1.1.0 (2015-08-12)
=================================

This is a mostly bugfix release with more solid support for Python 3.

Enhancements
------------

- Add ``Model.color()`` method.

Bugfixes
--------

- Remove ``loglmax`` from result of `sncosmo.nest_lc()`, which was not
  officially documented or supported. Use ``np.max(res.logl)`` instead.
- Fixed bug that caused non-reproducible behavior in
  `sncosmo.nest_lc()` even when ``np.random.seed()`` was called
  directly beforehand.
- Fixed file I/O problems on Python 3 related to string encoding.
- Fixed problem with SDSS bandpasses being stored as integers internally,
  preventing them from being used with models with dust.
- Fix problem where built-in source name and version strings were being
  dropped.
- Minor doc fixes.



What's new in v1.0 (2015-02-23)
===============================

API changes
-----------

- The API of ``mcmc_lc`` has changed significantly (the function was marked
  experimental in previous release).
- **[DEPRECATION]** In result of ``fit_lc``, ``res.cov_names`` changed to
  ``res.vparam_names``.
- **[DEPRECATION]** In result of ``nest_lc``, ``res.param_names``
  changed to ``res.vparam_names``. This is for compatibility between
  the results of ``fit_lc`` and ``nest_lc``. [#30]
- **[DEPRECATION]** Deprecate ``flatten`` keyword argument in ``fit_lc()`` in
  favor of explicit use of ``flatten_result()`` function.


Enhancements
------------

- Many new built-in models.
- Many new built-in bandpasses.
- New remote data fetching system.
- SALT2 model covariance available via ``Model.bandfluxcov()`` method and
  ``modelcov=True`` keyword argument passed to ``fit_lc``.
- New simulation function, ``zdist``, generates a distribution of redshifts
  given a volumetric rate function and cosmology.
- New simulation function, ``realize_lcs``, simulates light curve data given a
  model, parameters, and observations.
- Add color-related keyword arguments to ``plot_lc()``.
- Add ``tighten_ylim`` keyword argument to ``plot_lc()``.
- Add ``chisq()`` function and use internally in ``fit_lc()``.
- Add ``SFD98Map`` class for dealing with SFD (1998) dust maps persistently so
  that the underlying FITS files are opened only once. 
- Update ``get_ebv_from_map()`` to work with new SkyCoord class in
  ``astropy.coordinates`` available in astropy v0.3 onward. Previously, this
  function did not work with astropy v0.4.x (where older coordinates classes
  had been removed).
- Update to new configuration system available in astropy v0.4 onward.
  This makes this release incompatible with astropy versions less than
  0.4.
- Now compatible with Python 3.
- Increased test coverage.
- Numerous minor bugfixes.


What's new in v0.4 (2014-03-26)
===============================

This is non-backwards-compatible release, due to changes in the way
models are defined. These changes were made after feedback on the initial
design.

The most major change is a new central class ``Model`` used throughout
the pacakge. A ``Model`` instance encompasses a ``Source`` and zero or
more ``PropagationEffect`` instances. This is so that different
source models (e.g., SALT2 or spectral time series models) can be
combined with arbitrary dust models. The best way to think about this
is ``Source`` and ``PropagationEffect`` define the rest-frame behavior
of a SN and dust, and a ``Model`` puts these together to determine the
observer-frame behavior.

- New classes

  - ``sncosmo.Model``: new main container class
  - ``sncosmo.Source``: replaces existing ``Model``
  - ``sncosmo.TimeSeriesSource``: replaces existing ``TimeSeriesModel``
  - ``sncosmo.StretchSource``: replaces existing ``StretchModel``
  - ``sncosmo.SALT2Source``: replaces existing ``SALT2Model``
  - ``sncosmo.PropagationEffect``
  - ``sncosmo.CCM89Dust``
  - ``sncosmo.OD94Dust``
  - ``sncosmo.F99Dust``

- New public functions

  - ``sncosmo.read_griddata_ascii``: Read file with ``phase wave flux`` rows
  - ``sncosmo.read_griddata_fits``
  - ``sncosmo.write_griddata_fits``
  - ``sncosmo.nest_lc``: Nested sampling parameter estimation of SN model
  - ``sncosmo.simulate_vol`` (EXPERIMENTAL): simulation convenience function.

- Built-ins

  - updated SALT2 model URLs
  - added SALT2 version 2.4 (Betoule et al 2014)

- Improvements to ``sncosmo.plot_lc``: flexibility and layout

- Many bugfixes


What's new in v0.3 (2013-11-07)
===============================

This is a release with mostly bugfixes but a few new features,
designed to be backwards compatible with v0.2.0 ahead of API changes
coming in the next version.

Enhancements
------------

* New Functions

  - ``sncosmo.get_ebv_from_map``: E(B-V) at given coordinates from SFD map. 
  - ``sncosmo.read_snana_ascii``: Read SNANA ascii format files.
  - ``sncosmo.read_snana_fits``: Read SNANA FITS format files.
  - ``sncosmo.read_snana_simlib``: Read SNANA ascii "SIMLIB" files.

* registry is now case-independent. All of the following now work::

      sncosmo.get_magsystem('AB')
      sncosmo.get_magsystem('Ab')
      sncsomo.get_magsystem('ab')

* Photometric data can be unordered in time. Internally, the data are
  sorted before being used in fitting and typing.

* Numerous bugfixes.


What's new in v0.2 (2013-08-20)
===============================

Enhancements
------------

* Added SN 2011fe Nearby Supernova Factory data to built-in models as
  ``'2011fe'``

* Previously "experimental" functions now included:

  * ``sncosmo.fit_lc`` (previously ``sncosmo.fit_model``)
  * ``sncosmo.read_lc`` (previously ``sncosmo.readlc``)
  * ``sncosmo.write_lc`` (previously ``sncosmo.writelc``)
  * ``sncosmo.plot_lc`` (previously ``sncosmo.plotlc``)

* New functions:

  * ``sncosmo.load_example_data``: Example photometric data.
  * ``sncosmo.mcmc_lc``: Markov Chain Monte Carlo parameter estimation.
  * ``sncosmo.animate_model``: Model animation using matplotlib.animation.

* Fitting: ``sncosmo.fit_lc`` now uses the iminuit package for
  minimization by default. This requires the iminuit package to be
  installed, but the old minimizer (from scipy) can still be used by
  setting the keyword ``method='l-bfgs-b'``.

* Plotting: Ability to plot model synthetic photometry
  without observed data, using the syntax::

      >>> sncosmo.plot_lc(model=model, bands=['band1', 'band2'])

* Photometric data format: Photometric data format is now more
  flexible, allowing various names for table columns.

v0.1 (2013-07-15)
=================

Initial release.
