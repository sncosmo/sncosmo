# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Classes to represent spectral models of astronomical transients."""

import abc
import os
import math
import copy
from textwrap import dedent


import numpy as np
from astropy.utils import OrderedDict
from astropy.utils.misc import isiterable
from astropy import cosmology

from .utils import GridData1d, GridData2d, read_griddata, extinction_ratio_ccm
from .spectral import Spectrum, Bandpass, MagSystem, get_magsystem
from . import registry

__all__ = ['get_model', 'Model', 'TimeSeriesModel', 'StretchModel',
           'SALT2Model']

cosmology.set_current(cosmology.WMAP9)

def get_model(name, version=None):
    """Retrieve a model from the registry by name.

    Parameters
    ----------
    name : str
        Name of model in the registry.
    version : str, optional
        Version identifier for models with multiple versions. Default is
        `None` which corresponds to the latest, or only, version.
    """
    return registry.retrieve(Model, name, version=version)


class Model(object):
    """An abstract base class for transient models.
    
    A "transient model" in this case is the spectral time evolution
    of a source as a function of an arbitrary number of parameters.

    This is an abstract base class -- You can't create instances of
    this class. Instead, you must work with subclasses such as
    `TimeSeriesModel`. Subclasses must define (at minimum) `__init__()` and
    the private method `_model_flux_density()` 
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, name=None, version=None):
        self._params = OrderedDict()
        self._params['flux_scale'] = 1.
        self._params['absmag'] = None
        self._params['z'] = None
        self._refphase = 0.
        self._cosmo = cosmology.get_current()
        self._lumdist = None  # luminosity distance in Mpc
        self._name = name
        self._version = version

    def set(self, **params):
        """Set the parameters of the model.

        Parameters
        ----------
        flux_scale : float
            The internal model flux spectral density values will be multiplied
            by this factor. If set directly, `absmag` is set to `None`.
        absmag : tuple of (float, `sncosmo.Bandpass`, `sncosmo.MagSystem`)
            Set the `flux_scale` so that this is the absolute magnitude of
            the model spectrum at the reference rest-frame phase `refphase`.
            Cannot set at same time as `flux_scale`.
        z : float
            Redshift. If `None`, the model is in the rest-frame.
        """

        update_absmag = False

        if 'absmag' in params and 'flux_scale' in params:
            raise ValueError("cannot simulaneously set 'absmag' and"
                             "'flux_scale'")
        for key in params:
            if key not in self._params:
                raise ValueError("unknown parameter: '{}'".format(key))    
            if key == 'absmag':
                if params['absmag'] is None or len(params['absmag']) != 3:
                    raise ValueError("'absmag' must be (value, band, magsys)")
            if key not in ['z', 'flux_scale']:
                update_absmag = True
            self._params[key] = params[key]

        # If we set the redshift, update the luminosity distance.
        if 'z' in params:
            if params['z'] is None:
                self._lumdist = None
            elif self._cosmo is not None:
                self._lumdist = self._cosmo.luminosity_distance(params['z'])

        # If we set the flux scale, absolute mag becomes unknown.
        if 'flux_scale' in params:
            self._params['absmag'] = None
            update_absmag = False
        
        if update_absmag and self._params['absmag'] is not None:
            self._adjust_flux_scale_from_absmag()

    @abc.abstractmethod
    def _model_flux_density(self, phase=None, dispersion=None):
        """Return the model flux density (without any scaling or redshifting).
        """
        pass

    def _model_flux_density_error(self, phase=None, dispersion=None):
        """Return the model flux density error (without any scaling or
        redshifting. Return `None` if the error is undefined.
        """
        return None

    @property
    def params(self):
        """Dictionary of model parameters. (Read-only.) Use `set` method to
        set parameters."""
        return copy.copy(self._params)

    @property
    def cosmo(self):
        """Cosmology used to calculate luminosity distance."""
        return self._cosmo

    @cosmo.setter
    def cosmo(self, new_cosmology):
        if (new_cosmology is not None and 
            not isinstance(new_cosmology, cosmology.FLRW)):
            raise ValueError('cosmology must be None or '
                             'astropy.cosmology.FLRW instance.')
        self._cosmo = new_cosmology
        if self._cosmo is None:
            self._lumdist = None
        elif self._params['z'] is not None:
            self._lumdist = self._cosmo.luminosity_distance(self._params['z'])

    @property
    def refphase(self):
        """Phase at which absolute magnitude is evaluated.

        Typically set to the phase of maximum light in some band."""
        return self._refphase

    @refphase.setter
    def refphase(self, new_refphase):
        self._refphase = new_refphase

    def phases(self, restframe=False):
        """Return native phases of the model.

        Parameters
        ----------
        restframe : bool, optional
            If True, return rest-frame phases. Default is False, which
            corresponds to observer-frame phases (model phases
            multiplied by 1 + z).
        """

        if restframe or self._params['z'] is None:
            return self._phase
        else:
            return self._phase * (1. + self._params['z'])

    #TODO assumes wavelength
    def dispersion(self, restframe=False):
        """Return native dispersion grid of the model.

        Parameters
        ----------
        restframe : bool, optional
            If True, return rest-frame phases. Default is False, which
            corresponds to observer-frame phases (model phases
            multiplied by 1 + z).
        """

        if restframe or self._params['z'] is None:
            return self._dispersion
        else:
            return self._dispersion * (1. + self._params['z'])

    def flux_density(self, phase=None, dispersion=None, restframe=False):
        """The model flux spectral density.

        Parameters
        ----------
        phase : float or list_like, optional
            Model phase(s) in days. Phases are interpreted to be in the
            observer-frame unless `restframe` is True. If `None` (default)
            the native phases of the model are used.
        dispersion : float or list_like, optional
            Model dispersion values in Angstroms. Interpreted to be in the
            observer frame unless `restframe` is True. If `None` (default)
            the native dispersion of the model is used.
        restframe : bool, optional
            If True, return
        """

        if restframe or self._params['z'] is None:
            return (self._params['flux_scale'] *
                    self._model_flux_density(phase, dispersion))

        if phase is not None:
            phase /= (1. + self._params['z'])
        if dispersion is not None:
            dispersion /= (1. + self._params['z'])

        f = self._model_flux_density(phase, dispersion)

        # Apply rest-frame flux scaling and redshift
        factor = self._params['flux_scale'] / (1. + self._params['z'])
        if self._lumdist is not None:
            factor *= (1.e-5 / self._lumdist) ** 2

        f *= factor
        return f

    def flux_density_error(self, phase=None, dispersion=None, restframe=False):
        """The error on the model flux spectral density."""

        if restframe or self._params['z'] is None:
            f = self._model_flux_density_error(phase, dispersion)
            if f is None: return f
            return self._params['flux_scale'] * f

        if phase is not None:
            phase /= (1. + self._params['z'])
        if dispersion is not None:
            dispersion /= (1. + self._params['z'])

        f = self._model_flux_density_error(phase, dispersion)
        if f is None: return f

        # Apply rest-frame flux scaling, redshift factor, and lum distance.
        factor = self._params['flux_scale'] / (1. + self._params['z'])
        if self._lumdist is not None:
            factor *= (1.e-5 / self._lumdist) ** 2

        f *= factor
        return f

    def spectrum(self, phase, dispersion=None, restframe=False, 
                 include_error=False):
        """Return a `Spectrum` generated from the model at the given phase.

        Parameters
        ----------
        phase : float
            Model phase in days. Phase is interpreted to be in the
            observer frame unless `restframe` is True.
        dispersion : float or list_like, optional
            Model dispersion values in Angstroms. Interpreted to be in the
            observer frame unless `restframe` is True. If `None` (default)
            the native dispersion of the model is used.
        restframe : bool, optional
        include_error : bool, optional

        Returns
        -------
        spectrum : `sncosmo.Spectrum`

        Notes
        -----
        This is a convenience method for constructing a Spectrum.
        The spectrum is constructed by calling the `dispersion` and 
        `flux_density` (and optionally, `flux_density_error`) methods in
        the appropriate frame.
        """

        f = self.flux_density(phase, dispersion=dispersion,
                              restframe=restframe)

        if include_error:
            fe = self.flux_density_error(phase, dispersion=dispersion,
                                         restframe=restframe)
        else:
            fe = None

        if restframe:
            z = None
        else:
            z = self._params['z']

        if dispersion is None:
            dispersion = self.dispersion(restframe=restframe)
        return Spectrum(dispersion, f, error=fe, z=z)

    def flux(self, phase, band, restframe=False):
        """Flux (photons / cm^2 / s) at the given phase(s) through the given 
        bandpass(es).

        Parameters
        ----------
        phase : float (or list_like)
            Phase(s) in days.
        band : str or `Bandpass` (or list_like)
            Bandpass or name of bandpass in registry.

        Returns
        -------
        flux : float or `~numpy.ndarray`
            float if both phase and band are not interables.
            `~numpy.ndarray` if either is an interable.
        """

        phase, band = np.broadcast_arrays(phase, band)
        ndim = phase.ndim
        phase = phase.ravel()
        band = band.ravel()

        result = np.empty(phase.shape, dtype=np.float)

        for i, ph, b in zip(range(len(phase)), phase, band):
            s = self.spectrum(ph, restframe=restframe)
            result[i] = s.flux(b)
            
        if ndim == 0:
            return result[0]
        return result
            
    def mag(self, phase, band, magsys, restframe=False):
        """Magnitude at the given phase(s) through the given 
        bandpass(es), and for the given magnitude system(s).

        Parameters
        ----------
        phase : float or list_like
            Phase(s) in days.
        band : `Bandpass` or str (or list_like)
            Bandpass or name of bandpass in registry.
        magsys : `MagSystem` or str (or list_like)
            MagSystem or name of MagSystem in registry.
        restframe : bool, optional

        Returns
        -------
        mag : float or `~numpy.ndarray`
            Magnitude for each item in phase, band, magsys.
            The return value is a float if all parameters are not interables.
            The return value is an `~numpy.ndarray` if any are interable.
        """

        phase, band, magsys = np.broadcast_arrays(phase, band, magsys)
        ndim = phase.ndim
        phase = phase.ravel()
        band = band.ravel()
        magsys = magsys.ravel()

        result = np.empty(phase.shape, dtype=np.float)
        for i, ph, b, ms in zip(range(len(phase)), phase, band, magsys):
            ms = get_magsystem(ms)
            s = self.spectrum(ph, restframe=restframe)
            f = s.flux(b)
            zpf = ms.zpflux(b)
            result[i] = -2.5 * np.log10(f / zpf)

        if ndim == 0:
            return result[0]
        return result

    def _adjust_flux_scale_from_absmag(self):
        """Determine the flux_scale that when applied to the model
        (with current parameters), will result in the desired absolute
        magnitude at the reference phase."""

        mag, band, magsys = self._params['absmag']
        self._params['flux_scale'] = 1.
        m_current = self.mag(self._refphase, band, magsys,
                             restframe=True)
        self._params['flux_scale'] = 10.**(0.4 * (m_current - mag))

    def __call__(self, **params):
        """Return a shallow copy of the model with parameters set.
        See `set` method for details on parameters.
        
        Returns
        -------
        model : `sncosmo.Model`
            Model instance with parameters set as requested.

        Examples
        --------
        
        >>> model = sncosmo.get_model('salt2')
        >>> sn = model(c=0.1, x1=0.5, absmag=(-19.3, 'bessellb', 'ab'),
        ...            z=0.68)
        >>> sn.magnitude(0., 'desg', 'ab')
        25.577096820065883
        >>> sn = model(c=0., x1=0., absmag=(-19.3, 'bessellb', 'ab'),
        ...            z=0.5)
        >>> sn.magnitude(0., 'desg', 'ab')
        23.73028907970092
        """

        m = copy.copy(self)
        m.set(**params)
        return m

    def __repr__(self):
        name = ''
        version = ''
        if self._name is not None:
            name = ' {0!r:s}'.format(self._name)
        if self._version is not None:
            version = ' version={0!r:s}'.format(self._version)
        return "<{0:s}{1:s}{2:s} at 0x{3:x}>".format(
            self.__class__.__name__, name, version, id(self))


    def __str__(self):
        result = """\
        Model class: {}
        Model name: {}
        Model version: {}
        Restframe phases: [{:.6g}, .., {:.6g}] days ({:d} points)
        Restframe dipsersion: [{:.6g}, .., {:.6g}] Angstroms ({:d} points) 
        Reference phase: {} days
        Cosmology: {}
        Current Parameters:
        """.format(
            self.__class__.__name__,
            self._name, self._version,
            self._phase[0], self._phase[-1], len(self._phase),
            self._dispersion[0], self._dispersion[-1],
            len(self._dispersion),
            self._refphase,
            self._cosmo)
        result = dedent(result)

        parameter_lines = []
        for key, val in self._params.iteritems():
            if key == 'absmag':
                continue

            line = '    {}={}'.format(key, val)
            if key == 'flux_scale':
                line += ' [ absmag=' + str(self._params['absmag']) + ' ]'
            parameter_lines.append(line)
        return result + '\n'.join(parameter_lines)


class TimeSeriesModel(Model):
    """A single-component spectral time series model.

    Parameters
    ----------
    phase : `~numpy.ndarray`
        Phases in days.
    dispersion : `~numpy.ndarray`
        Wavelengths in Angstroms.
    flux_density : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape `(num_phases, num_dispersion)`.
    flux_density_error : `~numpy.ndarray`, optional
        Model error on `flux_density`. Must have same shape as `flux_density`.
        Default is `None`.
    extinction_func : function, optional
        A function that accepts an array of wavelengths in Angstroms
        and returns an array representing the ratio of total extinction
        (in magnitudes) to the parameter `c` at each wavelength. Default
        is `sncosmo.utils.extinction_ratio_ccm`
    extinction_kwargs : dict
        A dictionary of keyword arguments to pass to `extinction_func`.
        Default is `None`.
    name : str, optional
        Name of the model. Default is `None`.
    version : str, optional
        Version of the model. Default is `None`.
    """

    def __init__(self, phase, dispersion, flux_density,
                 flux_density_error=None, name=None, version=None,
                 extinction_func=extinction_ratio_ccm,
                 extinction_kwargs=None):

        super(TimeSeriesModel, self).__init__(name, version)
        self._params['c'] = None

        self._phase = phase
        self._dispersion = dispersion
        self._model = GridData2d(phase, dispersion, flux_density)

        self.set_extinction_func(extinction_func, extinction_kwargs)

        if flux_density_error is None:
            self._modelerror = None
        else:
            self._modelerror = GridData2d(phase, dispersion, flux_density_error)

    def _model_flux_density(self, phase=None, dispersion=None):
        """Return the model flux density (without any scaling or redshifting).
        """
        if self._params['c'] is None:
            return self._model(phase, dispersion)
        
        dust_trans = self._dust_trans_base(dispersion) ** self._params['c']
        return dust_trans * self._model(phase, dispersion)

    def _model_flux_density_error(self, phase=None, dispersion=None):
        """Return the model flux density error (without any scaling or
        redshifting. Return `None` if the error is undefined.
        """
        if self._modelerror is None:
            return None

        if self._params['c'] is None:
            return self._modelerror(phase, dispersion)
        
        dust_trans = self._dust_trans_base(dispersion) ** self._params['c']
        return dust_trans * self._modelerror(phase, dispersion)

    def set_extinction_func(self, func, extra_params):
        """Set the extinction ratio function to use in the model.

        Parameters
        ----------
        func : function
            A function that accepts an array of wavelengths in Angstroms
            and returns the ratio of total extinction
            (in magnitudes) to the parameter `c` at each wavelength.
        extra_params : dict
            A dictionary of keyword arguments to pass to the function.
            Default is `None`.
        """
        
        # set these parameters for info
        self._extinction_func = func
        self._extinction_kwargs = extra_params

        if extra_params is None:
            ext_ratio = func(self._dispersion)
        else:
            ext_ratio = func(self._dispersion, **extra_params)

        # calculate extinction base values, so that
        # self._dust_trans_base ** c gives the dust transmission.
        dust_trans_base =  10. ** (-0.4 * ext_ratio)
        self._dust_trans_base = GridData1d(self._dispersion, dust_trans_base)


class StretchModel(TimeSeriesModel):
    """A single-component spectral time series model, that "stretches".

    Parameters
    ----------
    phase : `~numpy.ndarray`
        Phases in days.
    dispersion : `~numpy.ndarray`
        Wavelengths in Angstroms.
    flux_density : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape `(num_phases, num_dispersion)`.

    Other Parameters
    ----------------
    flux_density_error : `~numpy.ndarray`, optional
        Model error on `flux_density`. Must have same shape as `flux_density`.
        Default is `None`.
    extinction_ratio_func : function, optional
        A function that accepts an array of wavelengths in Angstroms
        and returns an array representing the ratio of total extinction
        (in magnitudes) to the parameter `c` at each wavelength. Default
        is `sncosmo.utils.extinction_ratio_ccm`
    extinction_ratio_kwargs : dict
        A dictionary of keyword arguments to pass to `extinction_ratio_func`.
        Default is `None`.
    name : str, optional
        Name of the model. Default is `None`.
    version : str, optional
        Version of the model. Default is `None`.
    """
    def __init__(self, phase, dispersion, flux_density,
                 flux_density_error=None, name=None, version=None,
                 extinction_ratio_func=extinction_ratio_ccm,
                 extinction_ratio_kwargs=None):

        super(StretchModel, self).__init__(
            phase, dispersion, flux_density,
            flux_density_error=flux_density_error,
            name=name, version=version,
            extinction_ratio_func=extinction_ratio_func,
            extinction_ratio_kwargs=extinction_ratio_kwargs)
        
        self._params['s'] = None

    def _model_flux_density(self, phase=None, dispersion=None):
        """Return the model flux density (without any scaling or redshifting).
        """
        if phase is not None and self._params['s'] is not None:
            phase = phase / self._params['s']
        return super(StretchModel, self)._model_flux_density(phase, dispersion)

    def _model_flux_density_error(self, phase=None, dispersion=None):
        """Return the model flux density error (without any scaling or
        redshifting. Return `None` if the error is undefined.
        """
        if phase is not None and self._params['s'] is not None:
            phase = phase / self._params['s']
        return super(StretchModel, self)._model_flux_density_error(
            phase, dispersion)

    def _adjust_flux_scale_from_absmag(self, absmag):
        """Need to override this so that the refphase is applied correctly.
        We want refphase to refer to the *unstretched* phase. Otherwise the
        phase of peak will shift with s, if the peak is not at phase = 0."""

        s = self._params['s']
        self._params['s'] = None # temporarily remove stretch
        super(StretchModel, self)._adjust_flux_scale_from_absmag()
        self._params['s'] = s # put it back.


    def phases(self, restframe=False):
        """Return native phases of the model.

        Parameters
        ----------
        restframe : bool, optional
            If True, return rest-frame phases. Default is False, which
            corresponds to observer-frame phases (model phases
            multiplied by 1 + z).
        """

        s = self._params['s']
        z = self._params['z']

        if (restframe or z is None): 
            if s is None:
                return self._phase
            z = 0.
        if s is None:
            s = 1.

        return s * (1. + z) * self._phase



class SALT2Model(Model):
    """
    The SALT2 Type Ia supernova spectral timeseries model.

    Parameters
    ----------
    modeldir : str, optional
        Path to files containing model components. Each file should have
        the format:
 
            phase wavelength value

        on each line. If you want to give the absolute path to each file,
        set to `None`.
    m0file, m1file, v00file, v11file, v01file : str or fileobj, optional
        Filenames of various model components. Defaults are:

        * m0file = 'salt2_template_0.dat'
        * m1file = 'salt2_template_1.dat'
        * v00file = 'salt2_spec_variance_0.dat'
        * v11file = 'salt2_spec_variance_1.dat'
        * v01file = 'salt2_spec_covariance_01.dat'

    errscalefile : str, optional
        Name of error scale file, same format as model component files.
        The default is ``None``, which means that the error scale will
        not be applied in the ``fluxerr()`` method. This is only used for
        template versions 1.1 and 1.0, not 2.0+.

    Notes
    -----
    The phase and wavelength values of the various components don't necessarily
    need to match. (In the most recent salt2 model data, they do not all 
    match.) The phase and wavelength values of the first model component
    (in ``m0file``) are taken as the "native" sampling of the model, even
    though these values might require interpolation of the other model
    components.
    """

    def __init__(self, modeldir=None,
                 m0file='salt2_template_0.dat',
                 m1file='salt2_template_1.dat',
                 v00file='salt2_spec_variance_0.dat',
                 v11file='salt2_spec_variance_1.dat',
                 v01file='salt2_spec_covariance_01.dat',
                 errscalefile=None, name=None, version=None):

        super(SALT2Model, self).__init__(name, version)
        self._params['c'] = 0.
        self._params['x1'] = 0.
        self._model = {}

        components = ['M0', 'M1', 'V00', 'V11', 'V01', 'errscale']
        names_or_objs = [m0file, m1file, v00file, v11file, v01file,
                         errscalefile]

        # Make filenames into full paths.
        if modeldir is not None:
            for i in range(len(names_or_objs)):
                if (names_or_objs[i] is not None and
                    isinstance(names_or_objs[i], basestring)):
                    names_or_objs[i] = os.path.join(modeldir, names_or_objs[i])

        for component, name_or_obj in zip(components, names_or_objs):

            # If the filename is None, that component is left out of the model
            if name_or_obj is None: continue

            # Get the model component from the file
            phase, wavelength, values = read_griddata(name_or_obj)
            self._model[component] = GridData2d(phase, wavelength, values)

            # The "native" phases and wavelengths of the model are those
            # of the first model component.
            if component == 'M0':
                self._phase = phase
                self._dispersion = wavelength


    def _model_flux_density(self, phase=None, dispersion=None):
        """Return the model flux density (without any scaling or redshifting).
        """

        if phase is None:
            phase = self._phase
        if dispersion is None:
            dispersion = self._dispersion
        f0 = self._model['M0'](phase, dispersion)
        f1 = self._model['M1'](phase, dispersion)
        flux = f0 + self._params['x1'] * f1
        flux *= self._extinction(dispersion, self._params['c'])
        return flux

    def _model_flux_density_error(self, phase=None, dispersion=None):
        """Return the model flux density error (without any scaling or
        redshifting. Return `None` if the error is undefined.
        """
        if phase is None:
            phase = self._phase
        if dispersion is None:
            dispersion = self._dispersion
        phase = np.asarray(phase)
        dispersion = np.asarray(dispersion)

        v00 = self._model['V00'](phase, dispersion)
        v11 = self._model['V11'](phase, dispersion)
        v01 = self._model['V01'](phase, dispersion)

        #TODO apply x0 correctly
        # used to be sigma = x0 * np.sqrt(v00 + ...)
        x1 = self._params['x1']
        sigma = np.sqrt(v00 + x1 ** 2 * v11 + 2 * x1 * v01)
        sigma *= self._extinction(dispersion, self._params['c'])
        ### sigma *= 1e-12   #- Magic constant from SALT2 code
        
        if 'errscale' in self._model:
            sigma *= self._model['errscale'](phase, dispersion)
        
        # Hack adjustment to error (from SALT2 code)
        #TODO: figure out a way to do this.

        #if phase < -7. or phase > 12.:
        #    idx = (dispersion < 3400.)
        #    sigma[idx] *= 1000.
        
        return sigma


    def _extinction(self, wavelengths, c,
                    params=(-0.336646, +0.0484495)):
        """

        Notes
        -----
        From SALT2 code comments:

            ext = exp(color * constant * 
                      (l + params(0)*l^2 + params(1)*l^3 + ... ) /
                      (1 + params(0) + params(1) + ... ) )
                = exp(color * constant *  numerator / denominator )
                = exp(color * expo_term ) 
        """

        const = math.log(10)/2.5

        wB = 4302.57
        wV = 5428.55
        wr = (wavelengths - wB) / (wV - wB)

        numerator = 1.0 * wr
        denominator = 1.0

        wi = wr * wr
        for p in params:
            numerator += wi * p
            denominator += p
            wi *= wr

        return np.exp(c * const * numerator / denominator)
