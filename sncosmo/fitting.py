# Licensed under a 3-clause BSD style license - see LICENSE.rst

import copy
import math
import time
import warnings
from collections import OrderedDict

import numpy as np

from .bandpasses import Bandpass
from .photdata import photometric_data
from .utils import Interp1D, Result, ppf

__all__ = ['fit_lc', 'nest_lc', 'mcmc_lc', 'flatten_result', 'chisq']


class DataQualityError(Exception):
    pass


def generate_chisq(data, model, spectra, signature='iminuit', modelcov=False):
    """Define and return a chisq function for use in optimization.

    This function pre-computes and saves the inverse covariance matrix,
    making subsequent evaluations faster. The model covariance (if specified)
    is fixed at the time the chisq function is generated."""

    # precompute inverse covariance matrix
    if data is not None:
        cov = (np.diag(data.fluxerr**2) if data.fluxcov is None else
               data.fluxcov)
        if modelcov:
            _, mcov = model.bandfluxcov(data.band, data.time,
                                        zp=data.zp, zpsys=data.zpsys)
            cov = cov + mcov
        invcov = np.linalg.pinv(cov)

    # If we have spectra, build the covariance matrix for them individually.
    if spectra is not None:
        if modelcov:
            raise ValueError('modelcov not supported for spectra')

        spectra = np.atleast_1d(spectra)
        spectra_invcovs = []
        for spectrum in spectra:
            spectra_invcovs.append(np.linalg.pinv(spectrum.fluxcov))

    # iminuit expects each parameter to be a separate argument (including fixed
    # parameters)
    if signature == 'iminuit':
        def chisq(*parameters):
            # When a fit fails, iminuit sometimes calls the chisq function with
            # the parameters set to nan. This sometimes leads to a segfault in
            # sncosmo because the internal functions (specifically
            # BicubicInterpolator) aren't designed to handle that. See
            # https://github.com/sncosmo/sncosmo/issues/266 for details. For
            # now, return nan to minuit if it tries to set any parameter to
            # nan.
            if np.any(np.isnan(parameters)):
                return np.nan
            model.parameters = parameters

            full_chisq = 0.

            if data is not None:
                model_flux = model.bandflux(data.band, data.time,
                                            zp=data.zp, zpsys=data.zpsys)
                diff = data.flux - model_flux
                phot_chisq = np.dot(np.dot(diff, invcov), diff)
                full_chisq += phot_chisq

            if spectra is not None:
                for spectrum, spec_invcov in zip(spectra, spectra_invcovs):
                    sample_wave, sampling_matrix = \
                        spectrum.get_sampling_matrix()
                    sample_flux = model.flux(spectrum.time, sample_wave)
                    spec_model_flux = (
                        sampling_matrix.dot(sample_flux) /
                        sampling_matrix.dot(np.ones_like(sample_flux))
                    )
                    spec_diff = spectrum.flux - spec_model_flux
                    spec_chisq = spec_invcov.dot(spec_diff).dot(spec_diff)

                    full_chisq += spec_chisq

            return full_chisq
    else:
        raise ValueError("unknown signature: {!r}".format(signature))

    return chisq


def chisq(data, model, modelcov=False):
    """Calculate chisq statistic for the model, given the data.

    Parameters
    ----------
    model : `~sncosmo.Model`
    data : `~astropy.table.Table` or `~numpy.ndarray` or `dict`
        Table of photometric data. Must include certain columns.
        See the "Photometric Data" section of the documentation for
        required columns.
    modelcov : bool
        Include model covariance? Calls ``model.bandfluxcov`` method
        instead of ``model.bandflux``. The source in the model must therefore
        implement covariance.

    Returns
    -------
    chisq : float
    """
    data = photometric_data(data)
    data.sort_by_time()

    if data.fluxcov is None and not modelcov:
        mflux = model.bandflux(data.band, data.time,
                               zp=data.zp, zpsys=data.zpsys)
        return np.sum(((data.flux - mflux) / data.fluxerr)**2)

    else:
        # need to invert a covariance matrix
        cov = (np.diag(data.fluxerr**2) if data.fluxcov is None
               else data.fluxcov)
        if modelcov:
            mflux, mcov = model.bandfluxcov(data.band, data.time,
                                            zp=data.zp, zpsys=data.zpsys)
            cov = cov + mcov
        else:
            mflux = model.bandflux(data.band, data.time,
                                   zp=data.zp, zpsys=data.zpsys)
        invcov = np.linalg.pinv(cov)
        diff = data.flux - mflux
        return np.dot(np.dot(diff, invcov), diff)


def flatten_result(res):
    """Turn a result from fit_lc into a simple dictionary of key, value pairs.

    Useful when saving results to a text file table, where structures
    like a covariance matrix cannot be easily written to a single
    table row.

    Parameters
    ----------
    res : Result
        Result object from `~sncosmo.fit_lc`.

    Returns
    -------
    flatres : Result
        Flattened result. Keys are all strings, values are one of: float, int,
        string), suitable for saving to a text file.
    """

    flat = Result(success=(1 if res.success else 0),
                  ncall=res.ncall,
                  chisq=res.chisq,
                  ndof=res.ndof)

    # Parameters and uncertainties
    for i, n in enumerate(res.param_names):
        flat[n] = res.parameters[i]
        if res.errors is None:
            flat[n + '_err'] = float('nan')
        else:
            flat[n + '_err'] = res.errors.get(n, 0.)

    # Covariances.
    for n1 in res.param_names:
        for n2 in res.param_names:
            key = n1 + '_' + n2 + '_cov'
            if n1 not in res.cov_names or n2 not in res.cov_names:
                flat[key] = 0.
            elif res.covariance is None:
                flat[key] = float('nan')
            else:
                i = res.cov_names.index(n1)
                j = res.cov_names.index(n2)
                flat[key] = res.covariance[i, j]

    return flat


def _mask_bands(data, model, z_bounds=None):
    if z_bounds is None:
        return model.bandoverlap(data.band)
    else:
        return np.all(model.bandoverlap(data.band, z=z_bounds), axis=1)


def _warn_dropped_bands(data, mask):
    """Warn that we are dropping some bands from the data:"""
    drop_bands = [(b.name if b.name is not None else repr(b))
                  for b in set(data.band[np.invert(mask)])]
    warnings.warn("Dropping following bands from data: " +
                  ", ".join(drop_bands) +
                  "(out of model wavelength range)", RuntimeWarning)


def cut_bands(data, model, z_bounds=None, warn=True):
    mask = _mask_bands(data, model, z_bounds=z_bounds)

    if not np.all(mask):

        # Fail if there are no overlapping bands whatsoever.
        if not np.any(mask):
            raise RuntimeError('No bands in data overlap the model.')

        if warn:
            _warn_dropped_bands(data, mask)

        data = data[mask]

    return data, mask


def t0_bounds(data, model, spectra=None):
    """Determine bounds on t0 parameter of the model.

    The lower bound is such that the latest model time is equal to the
    earliest data time. The upper bound is such that the earliest
    model time is equal to the latest data time."""

    times = []

    if data is not None:
        times.append(data.time)

    if spectra is not None:
        for spec in np.atleast_1d(spectra):
            times.append(spec.time)

    times = np.hstack(times)

    return (model.get('t0') + np.min(times) - model.maxtime(),
            model.get('t0') + np.max(times) - model.mintime())


def _guess_t0_and_amplitude_photometry(data, model, minsnr):
    """Guess t0 and amplitude of the model from photometry."""

    # get data above the signal-to-noise ratio cut
    significant_data = data[(data.flux / data.fluxerr) > minsnr]
    if len(significant_data) == 0:
        raise DataQualityError('No data points with S/N > {0}. Initial '
                               'guessing failed.'.format(minsnr))

    # grid on which to evaluate model light curve
    timegrid = np.linspace(model.mintime(), model.maxtime(),
                           int(model.maxtime() - model.mintime() + 1))

    # get data flux on a consistent scale in order to compare to model
    # flux light curve.
    norm_flux = significant_data.normalized_flux(zp=25., zpsys='ab')

    model_lc = {}
    data_flux = {}
    data_time = {}
    for band in set(significant_data.band):
        model_lc[band] = (
            model.bandflux(band, timegrid, zp=25., zpsys='ab') /
            model.parameters[2])
        mask = significant_data.band == band
        data_flux[band] = norm_flux[mask]
        data_time[band] = significant_data.time[mask]

    if len(model_lc) == 0:
        raise DataQualityError('No data points with S/N > {0}. Initial '
                               'guessing failed.'.format(minsnr))

    # find band with biggest ratio of maximum data flux to maximum model flux
    maxratio = float("-inf")
    maxband = None
    for band in model_lc:
        ratio = np.max(data_flux[band]) / np.max(model_lc[band])
        if ratio > maxratio:
            maxratio = ratio
            maxband = band

    # amplitude guess is the largest ratio
    amplitude = abs(maxratio)

    # time guess is time of max in the band with the biggest ratio
    data_tmax = data_time[maxband][np.argmax(data_flux[maxband])]
    model_tmax = timegrid[np.argmax(model_lc[maxband])]
    t0 = model.get('t0') + data_tmax - model_tmax

    return t0, amplitude


def _guess_t0_and_amplitude_spectra(spectra, model, minsnr):
    """Guess t0 and amplitude of the model from spectra.

    The spectra don't necessarily have the same binning which makes this
    challenging. To handle this, we synthesize photometry in a range of
    different synthetic filters. We then call
    `_guess_t0_and_amplitude_photometry` on this photometry.
    """

    # Build a set of bands to use for synthetic photometry.
    target_band_width = 500     # Angstroms
    minwave = model.minwave()
    maxwave = model.maxwave()
    band_count = int(math.ceil((maxwave - minwave) / target_band_width))
    band_edges = np.linspace(minwave, maxwave, band_count+1)
    band_starts = band_edges[:-1]
    band_ends = band_edges[1:]

    bandpasses = np.array([Bandpass([start, end], [1., 1.]) for start, end
                           in zip(band_starts, band_ends)])

    all_bands = []
    all_fluxes = []
    all_fluxerrs = []
    all_times = []

    for spectrum in np.atleast_1d(spectra):
        # Find the bandpasses that overlap this spectrum.
        band_mask = ((spectrum.bin_edges[0] <= band_starts)
                     & (spectrum.bin_edges[-1] >= band_ends))
        spec_bands = bandpasses[band_mask]

        spec_flux, spec_fluxcov = spectrum.bandfluxcov(spec_bands, zp=25.,
                                                       zpsys='ab')
        spec_fluxerr = np.sqrt(np.diag(spec_fluxcov))

        all_bands.extend(spec_bands)
        all_fluxes.extend(spec_flux)
        all_fluxerrs.extend(spec_fluxerr)
        all_times.extend(np.ones_like(spec_flux) * spectrum.time)

    photometry = photometric_data({
        'band': all_bands,
        'flux': all_fluxes,
        'fluxerr': all_fluxerrs,
        'time': all_times,
        'zp': [25.] * len(all_fluxes),
        'zpsys': ['ab'] * len(all_fluxes),
    })

    # Sort the photometry by time
    photometry = photometry[np.argsort(photometry.time)]

    return _guess_t0_and_amplitude_photometry(photometry, model, minsnr)


def guess_t0_and_amplitude(data, model, minsnr, spectra=None):
    """Guess t0 and amplitude of the model based on the data.

    If we have photometry, we use it for the guessing. If not, we use all
    available spectra instead.
    """
    if data is not None:
        return _guess_t0_and_amplitude_photometry(data, model, minsnr)
    elif spectra is not None:
        return _guess_t0_and_amplitude_spectra(spectra, model, minsnr)
    else:
        raise ValueError('need either photometry or spectra to guess t0 and '
                         'amplitude.')


def _print_iminuit_params(names, kwargs):
    """Verbose printing of parameters to pass to Minuit"""
    for name in names:
        print(name, kwargs[name], 'step=', kwargs['error_' + name],
              end=" ")
        if 'limit_' + name in kwargs:
            print('bounds=', kwargs['limit_' + name], end=" ")
        print()


def _phase_and_wave_mask(data, t0, z, phase_range, wave_range):
    """Return a mask for the data based on an allowed rest-frame phase and/or
    wavelength range (given some t0 and z)."""
    if phase_range is not None:
        data_phase = (data.time - t0) / (1.0 + z)
        phase_mask = ((data_phase > phase_range[0]) &
                      (data_phase < phase_range[1]))

    if wave_range is not None:
        data_obswave = np.array([b.wave_eff for b in data.band])
        data_restwave = data_obswave / (1.0 + z)
        wave_mask = ((data_restwave > wave_range[0]) &
                     (data_restwave < wave_range[1]))

    if phase_range:
        if wave_range:
            return phase_mask & wave_mask
        return phase_mask
    if wave_range:
        return wave_mask
    return None


def fit_lc(data=None, model=None, vparam_names=[], bounds=None, spectra=None,
           method='minuit', guess_amplitude=True, guess_t0=True, guess_z=True,
           minsnr=5.0, modelcov=False, verbose=False, maxcall=10000,
           phase_range=None, wave_range=None, warn=True):
    """Fit model parameters to data by minimizing chi^2.

    Ths function defines a chi^2 to minimize, makes initial guesses for
    t0 and amplitude, then runs a minimizer.

    Parameters
    ----------
    data : `~astropy.table.Table` or `~numpy.ndarray` or `dict`
        Table of photometric data. Must include certain columns.
        See the "Photometric Data" section of the documentation for
        required columns.
    model : `~sncosmo.Model`
        The model to fit.
    vparam_names : list
        Model parameters to vary in the fit.
    bounds : `dict`, optional
        Bounded range for each parameter. Keys should be parameter
        names, values are tuples. If a bound is not given for some
        parameter, the parameter is unbounded. The exception is
        ``t0``: by default, the minimum bound is such that the latest
        phase of the model lines up with the earliest data point and
        the maximum bound is such that the earliest phase of the model
        lines up with the latest data point.
    spectra : `~sncosmo.Spectrum` or list of `~sncosmo.Spectrum` objects
        A list of spectra to include in the fit.
    guess_amplitude : bool, optional
        Whether or not to guess the amplitude from the data. If false, the
        current model amplitude is taken as the initial value. Only has an
        effect when fitting amplitude. Default is True.
    guess_t0 : bool, optional
        Whether or not to guess t0. Only has an effect when fitting t0.
        Default is True.
    guess_z : bool, optional
        Whether or not to guess z (redshift). Only has an effect when fitting
        redshift. Default is True.
    minsnr : float, optional
        When guessing amplitude and t0, only use data with signal-to-noise
        ratio (flux / fluxerr) greater than this value. Default is 5.
    method : {'minuit'}, optional
        Minimization method to use. Currently there is only one choice.
    modelcov : bool, optional
        Include model covariance when calculating chisq. Default is False.
        If true, the fit is performed multiple times until convergence.
    phase_range : (float, float), optional
        If given, discard data outside this range of phases. Note that
        **the definition of phase varies between models**: For example,
        phase=0 refers to explosion time in some models and time of peak
        B band flux in others.

        *New in version 1.5.0*

    wave_range : (float, float), optional
        If given, discard data with bandpass effective wavelengths outside
        this range.

        *New in version 1.5.0*

    verbose : bool, optional
        Print messages during fitting.
    warn : bool, optional
        Issue a warning when dropping bands outside the wavelength range of
        the model. Default is True.

        *New in version 1.5.0*

    Returns
    -------
    res : Result
        The optimization result represented as a ``Result`` object, which is
        a `dict` subclass with attribute access. Therefore, ``res.keys()``
        provides a list of the attributes. Attributes are:

        - ``success``: boolean describing whether fit succeeded.
        - ``message``: string with more information about exit status.
        - ``ncall``: number of function evaluations.
        - ``chisq``: minimum chi^2 value.
        - ``ndof``: number of degrees of freedom
          (len(data) - len(vparam_names)).
        - ``param_names``: same as ``model.param_names``.
        - ``parameters``: 1-d `~numpy.ndarray` of best-fit values
          (including fixed parameters) corresponding to ``param_names``.
        - ``vparam_names``: list of varied parameter names.
        - ``covariance``: 2-d `~numpy.ndarray` of parameter covariance;
          indicies correspond to order of ``vparam_names``.
        - ``errors``: OrderedDict of varied parameter uncertainties.
          Corresponds to square root of diagonal entries in covariance matrix.
        - ``nfit``: number of times the fit was performed. Can be greater than
          one when model covariance, phase range or wavelength range is used.
          *New in version 1.5.0.*
        - ``data_mask``: Boolean array the same length as data specifying
          whether each observation was used in the final fit.
          *New in version 1.5.0.*

    fitmodel : `~sncosmo.Model`
        A copy of the model with parameters set to best-fit values.

    Notes
    -----

    **t0 guess:** If ``t0`` is being fit and ``guess_t0=True``, the
    function will guess the initial starting point for ``t0`` based on
    the data. The guess is made as follows:

    * Evaluate the time and value of peak flux for the model in each band
      given the current model parameters.
    * Determine the data point with maximum flux in each band, for points
      with signal-to-noise ratio > ``minsnr`` (default is 5). If no points
      meet this criteria, the band is ignored (for the purpose of guessing
      only).
    * For each band, compare model's peak flux to the peak data point. Choose
      the band with the highest ratio of data / model.
    * Set ``t0`` so that the model's time of peak in the chosen band
      corresponds to the peak data point in this band.

    **amplitude guess:** If amplitude (assumed to be the first model parameter)
    is being fit and ``guess_amplitude=True``, the function will guess the
    initial starting point for the amplitude based on the data.

    **redshift guess:** If redshift (``z``) is being fit and ``guess_z=True``,
    the function will set the initial value of ``z`` to the average of the
    bounds on ``z``.

    Examples
    --------

    The `~sncosmo.flatten_result` function can be used to make the result
    a dictionary suitable for appending as rows of a table:

    >>> from astropy.table import Table
    >>> table_rows = []
    >>> for sn in sne:
    ...     res, fitmodel = sncosmo.fit_lc(
    ...          sn, model, ['t0', 'x0', 'x1', 'c'])
    ...     table_rows.append(flatten_result(res))
    >>> t = Table(table_rows)

    """

    if data is not None:
        # Standardize data
        data = photometric_data(data)

        # sort by time (some sources require this)
        # We keep track of indicies that sort the array because we want to be
        # able to report the indexes of the original data that were used in the
        # fit.
        if not np.all(np.ediff1d(data.time) >= 0.0):
            sortidx = np.argsort(data.time)
            data = data[sortidx]
        else:
            sortidx = None
    else:
        sortidx = None

    if spectra is not None:
        # Make sure that we have times for all of the spectra
        for spectrum in np.atleast_1d(spectra):
            if spectrum.time is None:
                raise TypeError("'time' must be set for each spectrum")

    if model is None:
        raise TypeError("missing required argument 'model'")

    # Make a copy of the model so we can modify it with impunity.
    model = copy.copy(model)

    # Check that vparam_names isn't empty and contains only parameters
    # known to the model.
    if len(vparam_names) == 0:
        raise ValueError("no parameters supplied")
    for s in vparam_names:
        if s not in model.param_names:
            raise ValueError("Parameter not in model: " + repr(s))

    # Order vparam_names the same way it is ordered in the model:
    vparam_names = [s for s in model.param_names if s in vparam_names]

    # initialize bounds
    bounds = copy.deepcopy(bounds) if bounds else {}

    # Check that 'z' is bounded (if it is going to be fit).
    if 'z' in vparam_names:
        if 'z' not in bounds or None in bounds['z']:
            raise ValueError('z must be bounded if fit.')
        if guess_z:
            model.set(z=sum(bounds['z']) / 2.)
        if model.get('z') < bounds['z'][0] or model.get('z') > bounds['z'][1]:
            raise ValueError('z out of range.')

    if data is not None:
        # Cut bands that are not allowed by the wavelength range of the model.
        fitdata, support_mask = cut_bands(data, model,
                                          z_bounds=bounds.get('z', None),
                                          warn=warn)
        # Initially this is the complete mask on data.
        data_mask = support_mask

        # Unique set of bands in data
        bands = set(fitdata.band.tolist())
    else:
        fitdata = None
        support_mask = None
        data_mask = None

    # Find t0 bounds to use, if not explicitly given
    if 't0' in vparam_names and 't0' not in bounds:
        bounds['t0'] = t0_bounds(fitdata, model, spectra)

    # Note that in the parameter guessing below, we assume that the source
    # amplitude is the 3rd parameter of the Model (1st parameter of the Source)

    # Turn off guessing if we're not fitting the parameter.
    if model.param_names[2] not in vparam_names:
        guess_amplitude = False
    if 't0' not in vparam_names:
        guess_t0 = False

    # Make guesses for t0 and amplitude.
    # (For now, we assume it is the 3rd parameter of the model.)
    if (guess_amplitude or guess_t0):
        t0, amplitude = guess_t0_and_amplitude(fitdata, model, minsnr, spectra)
        if guess_amplitude:
            model.parameters[2] = amplitude
        if guess_t0:
            model.set(t0=t0)

    if method == 'minuit':
        try:
            import iminuit
        except ImportError:
            raise ValueError("Minimization method 'minuit' requires the "
                             "iminuit package")

        # Set up keyword arguments to pass to Minuit initializer.
        kwargs = {}
        for name in model.param_names:
            kwargs[name] = model.get(name)  # Starting point.

            # Fix parameters not being varied in the fit.
            if name not in vparam_names:
                kwargs['fix_' + name] = True
                kwargs['error_' + name] = 0.
                continue

            # Bounds
            if name in bounds:
                if None in bounds[name]:
                    raise ValueError('one-sided bounds not allowed for '
                                     'minuit minimizer')
                kwargs['limit_' + name] = bounds[name]

            # Initial step size
            if name in bounds:
                step = 0.02 * (bounds[name][1] - bounds[name][0])
            elif model.get(name) != 0.:
                step = 0.1 * model.get(name)
            else:
                step = 1.
            kwargs['error_' + name] = step

        if verbose:
            print("Initial parameters:")
            _print_iminuit_params(vparam_names, kwargs)
            print()

        # run once with no model covariance, regardless of whether
        # modelcov=True
        fitchisq = generate_chisq(fitdata, model, spectra, signature='iminuit',
                                  modelcov=False)

        # count degrees of freedom
        ndof = 0
        if data is not None:
            ndof += len(fitdata)
        elif spectra is not None:
            for spectrum in np.atleast_1d(spectra):
                ndof += len(spectrum)
        ndof -= len(vparam_names)

        m = iminuit.Minuit(fitchisq, errordef=1.,
                           forced_parameters=model.param_names,
                           print_level=(1 if verbose >= 2 else 0),
                           throw_nan=True, **kwargs)
        d, l = m.migrad(ncall=maxcall)
        if verbose:
            print("{} function calls; {} dof.".format(d.nfcn, ndof))

        # numpy array of best-fit values (including fixed parameters).
        parameters = np.array([m.values[name] for name in model.param_names])
        model.parameters = parameters  # set model parameters to best fit.

        # Iterative Fitting

        if phase_range or wave_range:
            if spectra is not None:
                raise ValueError('phase_range and wave_range are not '
                                 'supported for spectra')
            range_mask = _phase_and_wave_mask(data, model.get('t0'),
                                              model.get('z'),
                                              phase_range, wave_range)
            data_mask = range_mask & support_mask

        # if model covariance, we need to re-run iteratively until convergence
        # if phase range is given, we need to rerun if there are any
        # masked points.
        refit = (modelcov or ((phase_range or wave_range) and
                              np.any(data_mask != support_mask)))

        nfit = 1
        while refit:
            # set new starting point to last point
            for name in vparam_names:
                kwargs[name] = model.get(name)

            if verbose:
                print("Initial parameters:")
                _print_iminuit_params(vparam_names, kwargs)
                print()

            # re-crop data based on ranges, if necessary
            if (phase_range or wave_range):
                fitdata = data[data_mask]

            # count degrees of freedom
            ndof = 0
            if data is not None:
                ndof += len(fitdata)
            elif spectra is not None:
                for spectrum in np.atleast_1d(spectra):
                    ndof += len(spectrum)
            ndof -= len(vparam_names)

            # generate chisq function based on new starting point
            fitchisq = generate_chisq(fitdata, model, spectra,
                                      signature='iminuit', modelcov=modelcov)

            m = iminuit.Minuit(fitchisq, errordef=1.,
                               forced_parameters=model.param_names,
                               print_level=(1 if verbose >= 2 else 0),
                               throw_nan=True, **kwargs)
            d, l = m.migrad(ncall=maxcall)

            if verbose:
                print("{} function calls; {} dof.".format(d.nfcn, ndof))

            parameters = np.array([m.values[name]
                                   for name in model.param_names])
            model.parameters = parameters
            nfit += 1

            refit = False
            # only consider refitting if we got a valid answer and we're under
            # the maximum number of iterations:
            if d.is_valid and nfit < 22:
                # recalculate data mask based on new t0, z
                if phase_range or wave_range:
                    old_data_mask = data_mask
                    range_mask = _phase_and_wave_mask(data, model.get('t0'),
                                                      model.get('z'),
                                                      phase_range, wave_range)
                    data_mask = support_mask & range_mask

                    # we'll refit if we changed any data
                    refit = np.any(data_mask != old_data_mask)

                # refit if *any* parameter changed by more than 10% of
                # statistical error bar
                if modelcov:
                    for name in vparam_names:
                        frac_change = (abs(m.values[name] - kwargs[name]) /
                                       m.errors[name])
                        refit = refit or frac_change > 0.1

        # Build a message.
        message = []
        if d.has_reached_call_limit:
            message.append('Reached call limit.')
        if d.hesse_failed:
            message.append('Hesse Failed.')
        if not d.has_covariance:
            message.append('No covariance.')
        elif not d.has_accurate_covar:  # iminuit docs wrong
            message.append('Covariance may not be accurate.')
        if not d.has_posdef_covar:  # iminuit docs wrong
            message.append('Covariance not positive definite.')
        if d.has_made_posdef_covar:
            message.append('Covariance forced positive definite.')
        if not d.has_valid_parameters:
            message.append('Parameter(s) value and/or error invalid.')
        if len(message) == 0:
            message.append('Minimization exited successfully.')
        # iminuit: m.np_matrix() doesn't work

        # Covariance matrix (only varied parameters) as numpy array.
        if m.covariance is None:
            covariance = None
        else:
            covariance = np.array([
                [m.covariance[(n1, n2)] for n1 in vparam_names]
                for n2 in vparam_names])

        # OrderedDict of errors
        if m.errors is None:
            errors = None
        else:
            errors = OrderedDict((name, m.errors[name])
                                 for name in vparam_names)

        # If we need to, unsort the mask so mask applies to input data
        if sortidx is not None:
            unsort_idx = np.argsort(sortidx)  # indicies that will unsort array
            data_mask = data_mask[unsort_idx]

        # Compile results
        res = Result(success=d.is_valid,
                     message=' '.join(message),
                     ncall=d.nfcn,
                     chisq=d.fval,
                     ndof=ndof,
                     param_names=model.param_names,
                     parameters=parameters,
                     vparam_names=vparam_names,
                     covariance=covariance,
                     errors=errors,
                     nfit=nfit,
                     data_mask=data_mask)

    else:
        raise ValueError("unknown method {0:r}".format(method))

    return res, model


def nest_lc(data, model, vparam_names, bounds, guess_amplitude_bound=False,
            minsnr=5., priors=None, ppfs=None, npoints=100, method='single',
            maxiter=None, maxcall=None, modelcov=False, rstate=None,
            verbose=False, warn=True, **kwargs):
    """Run nested sampling algorithm to estimate model parameters and evidence.

    Parameters
    ----------
    data : `~astropy.table.Table` or `~numpy.ndarray` or `dict`
        Table of photometric data. Must include certain columns.
        See the "Photometric Data" section of the documentation for
        required columns.
    model : `~sncosmo.Model`
        The model to fit.
    vparam_names : list
        Model parameters to vary in the fit.
    bounds : `dict`
        Bounded range for each parameter. Bounds must be given for
        each parameter, with the exception of ``t0``: by default, the
        minimum bound is such that the latest phase of the model lines
        up with the earliest data point and the maximum bound is such
        that the earliest phase of the model lines up with the latest
        data point.
    guess_amplitude_bound : bool, optional
        If true, bounds for the model's amplitude parameter are determined
        automatically based on the data and do not need to be included in
        `bounds`. The lower limit is set to zero and the upper limit is 10
        times the amplitude "guess" (which is based on the highest-flux
        data point in any band). Default is False.
    minsnr : float, optional
        Minimum signal-to-noise ratio of data points to use when guessing
        amplitude bound. Default is 5.
    priors : `dict`, optional
        Prior probability distribution function for each parameter. The keys
        should be parameter names and the values should be callables that
        accept a float. If a parameter is not in the dictionary, the prior
        defaults to a flat distribution between the bounds.
    ppfs : `dict`, optional
        Prior percent point function (inverse of the cumulative distribution
        function) for each parameter. If a parameter is in this dictionary,
        the ppf takes precedence over a prior pdf specified in ``priors``.
    npoints : int, optional
        Number of active samples to use. Increasing this value increases
        the accuracy (due to denser sampling) and also the time
        to solution.
    method : {'classic', 'single', 'multi'}, optional
        Method used to select new points. Choices are 'classic',
        single-ellipsoidal ('single'), multi-ellipsoidal ('multi'). Default
        is 'single'.
    maxiter : int, optional
        Maximum number of iterations. Iteration may stop earlier if
        termination condition is reached. Default is no limit.
    maxcall : int, optional
        Maximum number of likelihood evaluations. Iteration may stop earlier
        if termination condition is reached. Default is no limit.
    modelcov : bool, optional
        Include model covariance when calculating chisq. Default is False.
    rstate : `~numpy.random.RandomState`, optional
        RandomState instance. If not given, the global random state of the
        ``numpy.random`` module will be used.
    verbose : bool, optional
        Print running evidence sum on a single line.
    warn : bool, optional
        Issue warning when dropping bands outside the model range. Default is
        True.

        *New in version 1.5.0*

    Returns
    -------
    res : Result
        Attributes are:

        * ``niter``: total number of iterations
        * ``ncall``: total number of likelihood function calls
        * ``time``: time in seconds spent in iteration loop.
        * ``logz``: natural log of the Bayesian evidence Z.
        * ``logzerr``: estimate of uncertainty in logz (due to finite sampling)
        * ``h``: Bayesian information.
        * ``vparam_names``: list of parameter names varied.
        * ``samples``: 2-d `~numpy.ndarray`, shape is (nsamples, nparameters).
          Each row is the parameter values for a single sample. For example,
          ``samples[0, :]`` is the parameter values for the first sample.
        * ``logprior``: 1-d `~numpy.ndarray` (length=nsamples);
          log(prior volume) for each sample.
        * ``logl``: 1-d `~numpy.ndarray` (length=nsamples); log(likelihood)
          for each sample.
        * ``weights``: 1-d `~numpy.ndarray` (length=nsamples);
          Weight corresponding to each sample. The weight is proportional to
          the prior * likelihood for the sample.
        * ``parameters``: 1-d `~numpy.ndarray` of weighted-mean parameter
          values from samples (including fixed parameters). Order corresponds
          to ``model.param_names``.
        * ``covariance``: 2-d `~numpy.ndarray` of parameter covariance;
          indicies correspond to order of ``vparam_names``. Calculated from
          ``samples`` and ``weights``.
        * ``errors``: OrderedDict of varied parameter uncertainties.
          Corresponds to square root of diagonal entries in covariance matrix.
        * ``ndof``: Number of degrees of freedom (len(data) -
          len(vparam_names)).
        * ``bounds``: Dictionary of bounds on varied parameters (including
          any automatically determined bounds).
        * ``data_mask``: Boolean array the same length as data specifying
          whether each observation was used.
          *New in version 1.5.0.*

    estimated_model : `~sncosmo.Model`
        A copy of the model with parameters set to the values in
        ``res.parameters``.
    """

    try:
        import nestle
    except ImportError:
        raise ImportError("nest_lc() requires the nestle package.")

    # experimental parameters
    tied = kwargs.get("tied", None)

    data = photometric_data(data)

    # sort by time
    if not np.all(np.ediff1d(data.time) >= 0.0):
        sortidx = np.argsort(data.time)
        data = data[sortidx]
    else:
        sortidx = None

    model = copy.copy(model)
    bounds = copy.copy(bounds)  # need to copy this b/c we modify it below

    # Order vparam_names the same way it is ordered in the model:
    vparam_names = [s for s in model.param_names if s in vparam_names]

    # Drop data that the model doesn't cover.
    fitdata, data_mask = cut_bands(data, model,
                                   z_bounds=bounds.get('z', None),
                                   warn=warn)

    if guess_amplitude_bound:
        if model.param_names[2] not in vparam_names:
            raise ValueError("Amplitude bounds guessing enabled but "
                             "amplitude parameter {0!r} is not varied"
                             .format(model.param_names[2]))
        if model.param_names[2] in bounds:
            raise ValueError("cannot supply bounds for parameter {0!r}"
                             " when guess_amplitude_bound=True"
                             .format(model.param_names[2]))

        # If redshift is bounded, set model redshift to midpoint of bounds
        # when doing the guess.
        if 'z' in bounds:
            model.set(z=sum(bounds['z']) / 2.)
        _, amplitude = guess_t0_and_amplitude(fitdata, model, minsnr)
        bounds[model.param_names[2]] = (0., 10. * amplitude)

    # Find t0 bounds to use, if not explicitly given
    if 't0' in vparam_names and 't0' not in bounds:
        bounds['t0'] = t0_bounds(fitdata, model)

    if ppfs is None:
        ppfs = {}
    if tied is None:
        tied = {}

    # Convert bounds/priors combinations into ppfs
    if bounds is not None:
        for key, val in bounds.items():
            if key in ppfs:
                continue  # ppfs take priority over bounds/priors
            a, b = val
            if priors is not None and key in priors:
                # solve ppf at discrete points and return interpolating
                # function
                x_samples = np.linspace(0., 1., 101)
                ppf_samples = ppf(priors[key], x_samples, a, b)
                f = Interp1D(0., 1., ppf_samples)
            else:
                f = Interp1D(0., 1., np.array([a, b]))
            ppfs[key] = f

    # NOTE: It is important that iparam_names is in the same order
    # every time, otherwise results will not be reproducible, even
    # with same random seed.  This is because iparam_names[i] is
    # matched to u[i] below and u will be in a reproducible order,
    # so iparam_names must also be.
    iparam_names = [key for key in vparam_names if key in ppfs]
    ppflist = [ppfs[key] for key in iparam_names]
    npdim = len(iparam_names)  # length of u
    ndim = len(vparam_names)  # length of v

    # Check that all param_names either have a direct prior or are tied.
    for name in vparam_names:
        if name in iparam_names:
            continue
        if name in tied:
            continue
        raise ValueError("Must supply ppf or bounds or tied for parameter '{}'"
                         .format(name))

    def prior_transform(u):
        d = {}
        for i in range(npdim):
            d[iparam_names[i]] = ppflist[i](u[i])
        v = np.empty(ndim, dtype=np.float)
        for i in range(ndim):
            key = vparam_names[i]
            if key in d:
                v[i] = d[key]
            else:
                v[i] = tied[key](d)
        return v

    # Indicies of the model parameters in vparam_names
    idx = np.array([model.param_names.index(name) for name in vparam_names])

    def loglike(parameters):
        model.parameters[idx] = parameters
        return -0.5 * chisq(fitdata, model, modelcov=modelcov)

    t0 = time.time()
    res = nestle.sample(loglike, prior_transform, ndim, npdim=npdim,
                        npoints=npoints, method=method, maxiter=maxiter,
                        maxcall=maxcall, rstate=rstate,
                        callback=(nestle.print_progress if verbose else None))
    elapsed = time.time() - t0

    # estimate parameters and covariance from samples
    vparameters, cov = nestle.mean_and_cov(res.samples, res.weights)

    # update model parameters to estimated ones.
    model.set(**dict(zip(vparam_names, vparameters)))

    # If we need to, unsort the mask so mask applies to input data
    if sortidx is not None:
        unsort_idx = np.argsort(sortidx)  # indicies that will unsort array
        data_mask = data_mask[unsort_idx]

    # `res` is a nestle.Result object. Collect result into a sncosmo.Result
    # object for consistency, and add more fields.
    res = Result(niter=res.niter,
                 ncall=res.ncall,
                 logz=res.logz,
                 logzerr=res.logzerr,
                 h=res.h,
                 samples=res.samples,
                 weights=res.weights,
                 logvol=res.logvol,
                 logl=res.logl,
                 vparam_names=copy.copy(vparam_names),
                 ndof=len(fitdata) - len(vparam_names),
                 bounds=bounds,
                 time=elapsed,
                 parameters=model.parameters.copy(),
                 covariance=cov,
                 errors=OrderedDict(zip(vparam_names,
                                        np.sqrt(np.diagonal(cov)))),
                 param_dict=OrderedDict(zip(model.param_names,
                                            model.parameters)),
                 data_mask=data_mask)

    return res, model


def mcmc_lc(data, model, vparam_names, bounds=None, priors=None,
            guess_amplitude=True, guess_t0=True, guess_z=True,
            minsnr=5., modelcov=False, nwalkers=10, nburn=200,
            nsamples=1000, sampler='ensemble', ntemps=4, thin=1,
            a=2.0, warn=True):
    """Run an MCMC chain to get model parameter samples.

    This is a convenience function around `emcee.EnsembleSampler` andx
    `emcee.PTSampler`. It defines the likelihood function and makes a
    heuristic guess at a good set of starting points for the
    walkers. It then runs the sampler, starting with a burn-in run.

    If you're not getting good results, you might want to try
    increasing the burn-in, increasing the walkers, or specifying a
    better starting position.  To get a better starting position, you
    could first run `~sncosmo.fit_lc`, then run this function with all
    ``guess_[name]`` keyword arguments set to False, so that the
    current model parameters are used as the starting point.

    Parameters
    ----------
    data : `~astropy.table.Table` or `~numpy.ndarray` or `dict`
        Table of photometric data. Must include certain columns.
        See the "Photometric Data" section of the documentation for
        required columns.
    model : `~sncosmo.Model`
        The model to fit.
    vparam_names : iterable
        Model parameters to vary.
    bounds : `dict`, optional
        Bounded range for each parameter. Keys should be parameter
        names, values are tuples. If a bound is not given for some
        parameter, the parameter is unbounded. The exception is
        ``t0``: by default, the minimum bound is such that the latest
        phase of the model lines up with the earliest data point and
        the maximum bound is such that the earliest phase of the model
        lines up with the latest data point.
    priors : `dict`, optional
        Prior probability functions. Keys are parameter names, values are
        functions that return probability given the parameter value.
        The default prior is a flat distribution.
    guess_amplitude : bool, optional
        Whether or not to guess the amplitude from the data. If false, the
        current model amplitude is taken as the initial value. Only has an
        effect when fitting amplitude. Default is True.
    guess_t0 : bool, optional
        Whether or not to guess t0. Only has an effect when fitting t0.
        Default is True.
    guess_z : bool, optional
        Whether or not to guess z (redshift). Only has an effect when fitting
        redshift. Default is True.
    minsnr : float, optional
        When guessing amplitude and t0, only use data with signal-to-noise
        ratio (flux / fluxerr) greater than this value. Default is 5.
    modelcov : bool, optional
        Include model covariance when calculating chisq. Default is False.
    nwalkers : int, optional
        Number of walkers in the sampler.
    nburn : int, optional
        Number of samples in burn-in phase.
    nsamples : int, optional
        Number of samples in production run.
    sampler: str, optional
        The kind of sampler to use. Currently 'ensemble' for
        `emcee.EnsembleSampler` and 'pt' for `emcee.PTSampler` are
        supported.
    ntemps : int, optional
        If `sampler == 'pt'` the number of temperatures to use for the
        parallel tempered sampler.
    thin : int, optional
        Factor by which to thin samples in production run. Output samples
        array will have (nsamples/thin) samples.
    a : float, optional
        Proposal scale parameter passed to the sampler.
    warn : bool, optional
        Issue a warning when dropping bands outside the wavelength range of
        the model. Default is True.

        *New in version 1.5.0*

    Returns
    -------
    res : Result
        Has the following attributes:

        * ``param_names``: All parameter names of model, including fixed.
        * ``parameters``: Model parameters, with varied parameters set to
          mean value in samples.
        * ``vparam_names``: Names of parameters varied. Order of parameters
          matches order of samples.
        * ``samples``: 2-d array with shape ``(N, len(vparam_names))``.
          Order of parameters in each row  matches order in
          ``res.vparam_names``.
        * ``covariance``: 2-d array giving covariance, measured from samples.
          Order corresponds to ``res.vparam_names``.
        * ``errors``: dictionary giving square root of diagonal of covariance
          matrix for varied parameters. Useful for ``plot_lc``.
        * ``mean_acceptance_fraction``: mean acceptance fraction for all
          walkers in the sampler.
        * ``ndof``: Number of degrees of freedom (len(data) -
          len(vparam_names)).
          *New in version 1.5.0.*
        * ``data_mask``: Boolean array the same length as data specifying
          whether each observation was used.
          *New in version 1.5.0.*

    est_model : `~sncosmo.Model`
        Copy of input model with varied parameters set to mean value in
        samples.

    """

    try:
        import emcee
    except ImportError:
        raise ImportError("mcmc_lc() requires the emcee package.")

    # Standardize and normalize data.
    data = photometric_data(data)

    # sort by time
    if not np.all(np.ediff1d(data.time) >= 0.0):
        sortidx = np.argsort(data.time)
        data = data[sortidx]
    else:
        sortidx = None

    # Make a copy of the model so we can modify it with impunity.
    model = copy.copy(model)

    bounds = copy.deepcopy(bounds) if bounds else {}
    if priors is None:
        priors = {}

    # Check that vparam_names isn't empty, check for unknown parameters.
    if len(vparam_names) == 0:
        raise ValueError("no parameters supplied")
    for names in (vparam_names, bounds, priors):
        for name in names:
            if name not in model.param_names:
                raise ValueError("Parameter not in model: " + repr(name))

    # Order vparam_names the same way it is ordered in the model:
    vparam_names = [s for s in model.param_names if s in vparam_names]
    ndim = len(vparam_names)

    # Check that 'z' is bounded (if it is going to be fit).
    if 'z' in vparam_names:
        if 'z' not in bounds or None in bounds['z']:
            raise ValueError('z must be bounded if allowed to vary.')
        if guess_z:
            model.set(z=sum(bounds['z']) / 2.)
        if model.get('z') < bounds['z'][0] or model.get('z') > bounds['z'][1]:
            raise ValueError('z out of range.')

    # Cut bands that are not allowed by the wavelength range of the model.
    fitdata, data_mask = cut_bands(data, model,
                                   z_bounds=bounds.get('z', None),
                                   warn=warn)

    # Find t0 bounds to use, if not explicitly given
    if 't0' in vparam_names and 't0' not in bounds:
        bounds['t0'] = t0_bounds(fitdata, model)

    # Note that in the parameter guessing below, we assume that the source
    # amplitude is the 3rd parameter of the Model (1st parameter of the Source)

    # Turn off guessing if we're not fitting the parameter.
    if model.param_names[2] not in vparam_names:
        guess_amplitude = False
    if 't0' not in vparam_names:
        guess_t0 = False

    # Make guesses for t0 and amplitude.
    # (we assume amplitude is the 3rd parameter of the model.)
    if guess_amplitude or guess_t0:
        t0, amplitude = guess_t0_and_amplitude(fitdata, model, minsnr)
        if guess_amplitude:
            model.parameters[2] = amplitude
        if guess_t0:
            model.set(t0=t0)

    # Indicies used in probability function.
    # modelidx: Indicies of model parameters corresponding to vparam_names.
    # idxbounds: tuples of (varied parameter index, low bound, high bound).
    # idxpriors: tuples of (varied parameter index, function).
    modelidx = np.array([model.param_names.index(k) for k in vparam_names])
    idxbounds = [(vparam_names.index(k), bounds[k][0], bounds[k][1])
                 for k in bounds]
    idxpriors = [(vparam_names.index(k), priors[k]) for k in priors]

    # Posterior function.
    def lnlike(parameters):
        for i, low, high in idxbounds:
            if not low < parameters[i] < high:
                return -np.inf

        model.parameters[modelidx] = parameters
        logp = -0.5 * chisq(fitdata, model, modelcov=modelcov)
        return logp

    def lnprior(parameters):
        logp = 0
        for i, func in idxpriors:
            logp += math.log(func(parameters[i]))
        return logp

    def lnprob(parameters):
        return lnprior(parameters) + lnlike(parameters)

    # Heuristic determination of walker initial positions: distribute
    # walkers uniformly over parameter space. If no bounds are
    # supplied for a given parameter, use a heuristically determined
    # scale.

    if sampler == 'pt':
        pos = np.empty((ndim, nwalkers, ntemps))
        for i, name in enumerate(vparam_names):
            if name in bounds:
                pos[i] = np.random.uniform(low=bounds[name][0],
                                           high=bounds[name][1],
                                           size=(nwalkers, ntemps))
            else:
                ctr = model.get(name)
                scale = np.abs(ctr)
                pos[i] = np.random.uniform(low=ctr-scale, high=ctr+scale,
                                           size=(nwalkers, ntemps))
        pos = np.swapaxes(pos, 0, 2)
        sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlike, lnprob, a=a)

    # Heuristic determination of walker initial positions: distribute
    # walkers in a symmetric gaussian ball, with heuristically
    # determined scale.

    elif sampler == 'ensemble':
        ctr = model.parameters[modelidx]
        scale = np.ones(ndim)
        for i, name in enumerate(vparam_names):
            if name in bounds:
                scale[i] = 0.0001 * (bounds[name][1] - bounds[name][0])
            elif model.get(name) != 0.:
                scale[i] = 0.01 * model.get(name)
            else:
                scale[i] = 0.1
        pos = ctr + scale * np.random.normal(size=(nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a=a)

    else:
        raise ValueError('Invalid sampler type. Currently "pt" '
                         'and "ensemble" are supported.')

    # Run the sampler.
    pos, prob, state = sampler.run_mcmc(pos, nburn)  # burn-in
    sampler.reset()
    sampler.run_mcmc(pos, nsamples, thin=thin)  # production run
    samples = sampler.flatchain.reshape(-1, ndim)

    # Summary statistics.
    vparameters = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=0)
    model.set(**dict(zip(vparam_names, vparameters)))
    errors = OrderedDict(zip(vparam_names, np.sqrt(np.diagonal(cov))))
    mean_acceptance_fraction = np.mean(sampler.acceptance_fraction)

    # If we need to, unsort the mask so mask applies to input data
    if sortidx is not None:
        unsort_idx = np.argsort(sortidx)  # indicies that will unsort array
        data_mask = data_mask[unsort_idx]

    res = Result(param_names=copy.copy(model.param_names),
                 parameters=model.parameters.copy(),
                 vparam_names=vparam_names,
                 samples=samples,
                 covariance=cov,
                 errors=errors,
                 ndof=len(fitdata) - len(vparam_names),
                 mean_acceptance_fraction=mean_acceptance_fraction,
                 data_mask=data_mask)

    return res, model
