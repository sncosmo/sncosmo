from astropy.io import ascii
from astropy import cosmology
from astropy.constants import cgs

#h_erg_s = cgs.h
#c_AA_s = cgs.c * 1.e8
h_erg_s = 6.626068e-27  # Planck constant (erg * s)
c_AA_s = 2.9979e+18  # Speed of light ( AA / sec)

cosmo = cosmology.LambdaCDM(H0=70., Om0=0.3, Ode0=0.7)

class Spectrum(object):
    """A spectrum.

    Parameters
    ----------
    wavelength : list_like
        Wavelength values, in angstroms
    flux : list_like
        Flux values, in units f_lambda (ergs/s/cm^2/AA)
    variance : list_like, optional
        Variance on flux values.
    z : float, optional
        Redshift of spectrum (default is `None`)
    dist : float, optional
        Luminosity distance in Mpc, used to adjust flux (default is `None`)
    meta : OrderedDict, optional
        Metadata.
    copy : bool, optional
        Copy input arrays.
    """

    @classmethod
    def read(cls, filename):
        """Read a spectrum from a text file.

        The file must be a two column table, optionally with a one line header.

        Parameters
        ----------
        filename : string

        Returns
        -------
        spectrum : Spectrum object
        """
        table = ascii.read(filename, names=['wavelength', 'flux'])
        return cls(table['wavelength'], table['flux'], copy=True)


    def __init__(self, wavelength, flux, z=None, dist=None, variance=None,
                 meta=None, copy=False):
        
        self.wavelength = np.array(wavelength, copy=copy)
        self.flux = np.array(flux, copy=copy)
        if self.wavelength.shape != self.flux.shape:
            raise ValueError('shape of wavelength and flux must match')
        if self.wavelength.ndim != 1:
            raise ValueError('only 1-d arrays supported')
        self.z = z
        self.dist = dist
        if variance is not None:
            self.variance = np.array(variance, copy=copy)
            if self.wavelength.shape != self.variance.shape:
                raise ValueError('shape of wavelength and variance must match')
        else:
            self.variance = None
        self.meta = OrderedDict() if meta is None else deepcopy(meta)


    def synphot(self, band):
        """Perform synthentic photometry in a given bandpass.
      
        Parameters
        ----------
        band : Bandpass object
            Self explanatory.

        Returns
        -------
        flux : float
            Total flux in photons/sec/cm^2
        fluxerr : float
            Error on flux, only returned if spectrum.variance is not `None`
        """

        # If the bandpass is not fully inside the defined region of the spectrum
        # return None.
        if (band.wavelength[0] < spec.wavelength[0] or
            band.wavelength[-1] > spec.wavelength[-1]):
            return None

        # Get the spectrum index range to use
        idx = ((spec.wavelength > band.wavelength[0]) & 
               (spec.wavelength < band.wavelength[-1]))

        # Spectrum quantities in this wavelength range
        wave = spec.wavelength[idx]
        flux = spec.flux[idx]
        binwidth = np.gradient(wave) # Width of each bin

        # Interpolate bandpass transmission to these wavelength values
        trans = np.interp(wave, band.wavelength, band.transmission)

        # Convert flux from erg/s/cm^2/AA to photons/s/cm^2/AA
        factor = (wave / (h_erg_s * c_AA_s))
        flux *= factor

        # Get total erg/s/cm^2
        totflux = np.sum(flux * trans * binwidth)

        if spec.variance is None:
            return totflux

        else:
            var = spec.variance[idx]
            var *= factor ** 2  # Convert from erg/s/cm^2/AA to photons/s/cm^2/AA
            totvar = np.sum(var * trans ** 2 * binwidth) # Total variance
            return totflux, np.sqrt(totvar)


    def redshifted_to(self, z, dist=None, adjust_flux=False, cosmo=None):
        """Return a new Spectrum object at a new redshift.

        The current redshift must be defined (spec.z cannot be `None`).

        Parameters
        ----------
        z : float
            A factor of (1 + z) / (1 + self.z) is applied to the wavelength. 
            The inverse factor is applied to the flux so that the bolometric
            flux remains the same.
        dist : 

        dist_in OR zdist_in : float
            Input distance in Mpc or input redshift (> 0). Used to adjust
            bolometric flux, as F_out = F_in * (D_in / D_out) ** 2,
            where D is luminosity distance. D is given by dist_in or calculated
            for the redshift zdist_in. 
        dist_out OR zdist_out :float
            Output distance in Mpc or redshift (> 0). Used to adjust bolometric
            flux. See above.

        Returns
        -------
        spec : Spectrum object
            A new spectrum object at redshift z.
        """

        # Shift wavelengths, adjust flux so that bolometric flux
        # remains constant.
        factor =  (1. + z_out) / (1. + z_in)
        new_wave = spec.wavelength * factor
        new_flux = spec.flux / factor
        if spec.variance is not None:
            new_var = spec.variance / factor ** 2
        else:
            new_var = None

        # Check flux distance inputs
        if (dist_in is not None) and (zdist_in is not None):
            raise ValueError("cannot specify both zdist_in and dist_in")
        if zdist_in is not None:
            if zdist_in <= 0.:
                raise ValueError("zdist_in must be greater than 0")
            dist_in = cosmo.luminosity_distance(zdist_in)

        # Check flux distance outputs
        if (dist_out is not None) and (zdist_out is not None):
            raise ValueError("cannot specify both zdist_out and dist_out")
        if zdist_out is not None:
            if zdist_out <= 0.:
                raise ValueError("zdist_out must be greater than 0")
            dist_out = cosmo.luminosity_distance(zdist_out)

        # Check for only one being specified.
        if (((dist_in is None) and (dist_out is not None)) or
            ((dist_in is not None) and (dist_out is None))):
            raise ValueError("Only input or only output flux distance specified")

        # If both are specified, adjust the flux.
        if dist_in is not None and dist_out is not None:
            factor = (dist_in / dist_out) ** 2
            new_flux *= factor
            if new_var is not None:
                new_var *= factor ** 2

        return Spectrum(new_wave, new_flux, variance=new_var, meta=spec.meta)
