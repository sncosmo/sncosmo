*******
Spectra
*******

Spectroscopic observations are supported in `sncosmo` with the `~sncosmo.Spectrum`
class. A spectrum object can be created from a list of wavelengths and flux values:

.. code:: python

    >>> wave = np.arange(3000, 9000)
    >>> model = sncosmo.Model(source='hsiao-subsampled')
    >>> model.set(z=0.1, amplitude=1e-5, t0=56000.)
    >>> sample_time = model['t0'] + 2.
    >>> flux = model.flux(time=sample_time, wave=wave)

    >>> spectrum = sncosmo.Spectrum(wave, flux, time=sample_time)

By default, the wavelengths are assumed to be in angstroms and the flux values as a
spectral flux density (erg / s / cm^2 / A).

Uncertainties
-------------

A spectrum can have associated uncertainties. This can be either uncorrelated
uncertainties for each spectral element:

.. code:: python

    >>> fluxerr = 0.1 * flux
    >>> spectrum = sncosmo.Spectrum(wave, flux, fluxerr, time=sample_time)

or a full covariance matrix:

.. code:: python

    >>> fluxcov = np.diag(fluxerr**2) + 1e-5 * np.max(flux)**2
    >>> spectrum = sncosmo.Spectrum(wave, flux, fluxcov=fluxcov, time=sample_time)

All operations will take these uncertanties into account.

Synthetic photometry
--------------------

Synthetic photometry can be calculated on a spectrum using any of the bandpasses
available in sncosmo:

.. code:: python

    >>> spectrum.bandflux('sdssg')
    6.336874516839506

Synthetic photometry can be calculated on multiple bands simultaneously:

.. code:: python

    >>> spectrum.bandflux(['sdssg', 'sdssr', 'sdssi'])
    array([6.33687452, 5.33348909, 2.86301428])

If a zeropoint and magnitude system are specified, then the bandflux is returned in that
system (otherwise it is in photons / s / cm^2 by default).

.. code:: python

    >>> spectrum.bandflux(['sdssg', 'sdssr', 'sdssi'], zp=25., zpsys='ab')
    array([115932.40764299, 108077.89453051,  79602.18806937])

Optionally, the full covariance matrix between the bandfluxes can also be calculated:

.. code:: python

    >>> spectrum.bandfluxcov(['sdssg', 'sdssr', 'sdssi'], zp=25., zpsys='ab')
    (array([115932.40764299, 108077.89453051,  79602.18806937]),
     array([[ 436626.04457069,  574161.2530974 ,  845566.99897734],
            [ 574161.2530974 , 1088219.30397941, 1460106.55463119],
            [ 845566.99897734, 1460106.55463119, 2203060.35600117]]))

A band magnitude can be evaluated in a specific magnitude system:

.. code:: python

    >>> spectrum.bandmag(['sdssg', 'sdssr', 'sdssi'], magsys='ab')
    array([12.33948786, 12.41565781, 12.74768749])



Rebinning a spectrum
--------------------

A spectrum can be rebinned with arbitrary wavelength bins. This returns a new
`~sncosmo.Spectrum` object.

.. code:: python
    
    >>> binned_spectrum = spectrum.rebin(np.arange(3500, 6000, 100))

Rebinning introduces covariance between adjacent spectral elements if the bin edges
in the original spectrum don't line up with the bin edges in the rebinned spectrum. This
covariance is properly propagated.


Fitting with spectra
--------------------

Spectra can be used in fits. Any combination of spectra and photometry is allowed. For
example, to fit a single spectrum:

.. code:: python

    >>> model.set(z=0., amplitude=1., t0=0.)
    >>> sncosmo.fit_lc(model=model, spectra=binned_spectrum,
    ...                vparam_names=['amplitude', 't0', 'z'],
    ...                bounds={'z': (0., 0.3)})
    (      success: True
           message: 'Minimization exited successfully.'
             ncall: 86
             chisq: 1.1072097164403554e-05
              ndof: 22
       param_names: ['z', 't0', 'amplitude']
        parameters: array([9.99999822e-02, 5.60000000e+04, 9.99997056e-06])
      vparam_names: ['z', 't0', 'amplitude']
        covariance: array([[ 4.60410999e-08,  7.84028630e-06, -1.39915254e-12],
            [ 7.84028630e-06,  6.28193143e-03, -1.47231576e-09],
            [-1.39915254e-12, -1.47231576e-09,  2.86156520e-15]])
            errors: OrderedDict([('z', 0.0002145718167298541), ('t0', 0.07925860166142229), ('amplitude', 5.3493599593407034e-08)])
              nfit: 1
         data_mask: None,
     <sncosmo.models.Model at 0x7fb8d498c110>)

Other valid signatures are:

.. code:: python

    # photometry only
    >>> sncosmo.fit_lc(photometry, model, ...)

    # a single spectrum
    >>> sncosmo.fit_lc(model=model, spectra=spectrum, ...)

    # multiple spectra
    >>> sncosmo.fit_lc(model=model, spectra=[spec_1, spec_2], ...)

    # spectra and photometry simultaneously
    >>> sncosmo.fit_lc(photometry, model, spectra=[spec_1, spec_2], ...)
