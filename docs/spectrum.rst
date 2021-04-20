*******
Spectra
*******

Spectroscopic observations are supported in `sncosmo` with the
`~sncosmo.Spectrum` class. A spectrum object can be created from a list of
wavelengths and flux values:

.. code:: python

    >>> wave, flux, fluxerr = sncosmo.load_example_spectrum_data()
    >>> spectrum = sncosmo.Spectrum(wave, flux)

By default, the wavelengths are assumed to be in angstroms and the flux
values as a spectral flux density (erg / s / cm^2 / A).

Uncertainties
-------------

A spectrum can have associated uncertainties. This can be either uncorrelated
uncertainties for each spectral element:

.. code:: python

    >>> spectrum = sncosmo.Spectrum(wave, flux, fluxerr)

or a full covariance matrix:

.. code:: python

    >>> fluxcov = np.diag(fluxerr**2) + 1e-5 * np.max(flux)**2
    >>> spectrum = sncosmo.Spectrum(wave, flux, fluxcov=fluxcov)

All operations will take these uncertanties into account.

Synthetic photometry
--------------------

Synthetic photometry can be calculated on a spectrum using any of the bandpasses
available in sncosmo:

.. code:: python

    >>> spectrum.bandflux('sdssg')
    6.417843339246818

Synthetic photometry can be calculated on multiple bands simultaneously:

.. code:: python

    >>> spectrum.bandflux(['sdssg', 'sdssr', 'sdssi'])
    array([6.41784334, 5.37683496, 2.8626649 ])

If a zeropoint and magnitude system are specified, then the bandflux is
returned in that system (otherwise it is in photons / s / cm^2 by default).

.. code:: python

    >>> spectrum.bandflux(['sdssg', 'sdssr', 'sdssi'], zp=25., zpsys='ab')
    array([117413.72315598, 108956.25581652,  79592.47424074])

Optionally, the full covariance matrix between the bandfluxes can also be
calculated:

.. code:: python

    >>> spectrum.bandfluxcov(['sdssg', 'sdssr', 'sdssi'], zp=25., zpsys='ab')
    (array([117413.72315598, 108956.25581652,  79592.47424074]),
     array([[1546972.78077853,  874550.34470817, 1287550.14818921],
             [ 874550.34470817, 2479812.26652077, 2226272.0481113 ],
             [1287550.14818921, 2226272.0481113 , 3805168.84485814]]))

A band magnitude can be evaluated in a specific magnitude system:

.. code:: python

    >>> spectrum.bandmag(['sdssg', 'sdssr', 'sdssi'], magsys='ab')
    array([12.32570285, 12.40686957, 12.74781999])



Rebinning a spectrum
--------------------

A spectrum can be rebinned with arbitrary wavelength bins. This returns a new
`~sncosmo.Spectrum` object.

.. code:: python
    
    >>> binned_spectrum = spectrum.rebin(np.arange(3500, 6000, 100))

Rebinning introduces covariance between adjacent spectral elements if the bin
edges in the original spectrum don't line up with the bin edges in the
rebinned spectrum. This covariance is properly propagated.


Fitting with spectra
--------------------

Spectra can be used in fits. Any combination of spectra and photometry is
allowed. To fit spectra, the times at which the spectra were taken must be
specified. For example, to fit a single spectrum:

.. code:: python

    # Create the spectrum object, and specify the time at which it was taken.
    >>> spectrum = sncosmo.Spectrum(wave, flux, fluxerr, time=20.)

    # Fit a model to the spectrum.
    >>> model = sncosmo.Model(source='hsiao-subsampled')
    >>> sncosmo.fit_lc(model=model, spectra=spectrum,
    ...                vparam_names=['amplitude', 't0', 'z'],
    ...                bounds={'z': (0., 0.3)})
    (      success: True
           message: 'Minimization exited successfully.'
             ncall: 108
             chisq: 576.7111360163605
              ndof: 597
       param_names: ['z', 't0', 'amplitude']
        parameters: array([9.96571945e-02, 1.80278503e+01, 1.00650322e-05])
      vparam_names: ['z', 't0', 'amplitude']
        covariance: array([[ 1.17946556e-07,  1.64336679e-05, -5.21279026e-12],
                           [ 1.64336679e-05,  1.70047614e-02, -4.60755668e-09],
                           [-5.21279026e-12, -4.60755668e-09,  2.91915780e-15]])
            errors: OrderedDict([('z', 0.00034343314287464677),
                                 ('t0', 0.13040215158608248), 
                                 ('amplitude', 5.4029230945864686e-08)])
              nfit: 1
         data_mask: None,
    <sncosmo.models.Model at 0x7fa30159a6d0>)

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
