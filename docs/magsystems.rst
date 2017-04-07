*****************
Magnitude Systems
*****************

SNCosmo has facilities for converting synthetic model photometry to
magnitudes in a variety of magnitude systems (or equivalently, scaling
fluxes to a given zeropoint in a given magnitude system). For example,
in the following code snippet, the string `'ab'` specifies that we
want magnitudes on the AB magnitude system::

  >>> model.bandmag('desr', 'ab', [54990., 55000., 55020.])

The string ``'ab'`` here refers to a built-in magnitude system
(``'vega'`` is another option). Behind the scenes magnitude systems
are represented with `~sncosmo.MagSystem` objects. As with
`~sncosmo.Bandpass` objects, most places in SNCosmo that require a
magnitude system can take either the name of a magnitude system in the
registry or an actual `~sncosmo.MagSystem` instance. You can access
these objects directly or create your own.

`~sncosmo.MagSystem` objects represent the spectral flux density
corresponding to magnitude zero in the given system and can be used to
convert physical fluxes (in photons/s/cm^2) to magnitudes. Here's an
example::

  >>> ab = sncosmo.get_magsystem('ab')
  >>> ab.zpbandflux('sdssg')
  546600.83408598113

This example gives the number of counts (in photons) when integrating
the AB spectrum (which happens to be F_nu = 3631 Jansky at all
wavelengths) through the SDSS g band. This works similarly for other
magnitude systems::

  >>> vega = sncosmo.get_magsystem('vega')
  >>> vega.zpbandflux('sdssg')
  597541.25707788975

You can see that the Vega spectrum is a bit brighter than the AB
spectrum in this particular bandpass. Therefore, SDSS *g* magnitudes
given in Vega will be larger than if given in AB.

There are convenience methods for converting an observed flux in a
bandpass to a magnitude::

  >>> ab.band_flux_to_mag(1., 'sdssg')
  14.344175725172901
  >>> ab.band_mag_to_flux(14.344175725172901, 'sdssg')
  0.99999999999999833

So, one count per second in this band is equivalent to an AB magnitude
of about 14.34.


"Composite" magnitude systems
-----------------------------

Sometimes, photometric data is reported in "magnitude systems" that
don't correspond directly to any spectrophotometric standard. One
example is "SDSS magnitudes" which are like AB magnitudes but with an
offset in each band. These are represented in SNCosmo with the
`~sncosmo.CompositeMagSystem` class. For example::

  >>> magsys = sncosmo.CompositeMagSystem(bands={'sdssg': ('ab', 0.01),
  ...                                            'sdssr': ('ab', 0.02)})


This defines a new magnitude system that knows about only two
bandpasses. In this magnitude system, an object with magnitude zero in
AB would have a magntide of 0.01 in SDSS *g* and 0.02 in SDSS
*r*. Indeed, you can see that the flux corresponding to magnitude zero
is slightly higher in this magnitude system than in AB::

  >>> magsys.zpbandflux('sdssr')
  502660.28545283229

  >>> ab.zpbandflux('sdssr')
  493485.70128115633

Since we've only defined the offsets for this magnitude system in a
couple bands, using other bandpasses results in an error::

  >>> magsys.zpbandflux('bessellb')
  ValueError: band not defined in composite magnitude system
