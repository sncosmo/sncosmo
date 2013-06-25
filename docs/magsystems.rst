*****************
Magnitude Systems
*****************

A `sncosmo.MagSystem` object represents an astrophysical magnitude system
and can be used to convert physical fluxes to magnitudes (in that system). 
For example,

    >>> abmagsys = sncosmo.get_magsystem('ab')
    >>> abmagsys.zpbandflux('bessellb') # flux of a source with AB mag = 0 in given bandpass
    1199275.6325987775

Fluxes are in photons/s/cm^2.

"Local" Magnitude Systems
-------------------------

Make the magnitude system a "local magnitude system" by setting offsets.

    >>> sdss = sncosmo.ABMagSystem(refmags={'sdssg': 0.02, 'sdssr':0.03})

"SDSS" magnitudes now differ from the AB system in that an object of
magnitude 0 in the AB system will have magnitude 0.02 the "SDSS"
magnitude system (for the 'sdssg' band). That is,
`sdssg(SDSS) = sdssg(AB) + 0.02`

You can also set the reference magnitudes after the fact:

    >>> sdss = sncosmo.get_magsystem('ab')  # start with an AB system
    >>> sdss.refmags = {'sdssg': 0.02, 'sdssr':0.03}  # add offsets

But, you can't assign to refmags once created.

    >>> sdss.refmags['sdssi'] = 0.04  # doesn't do anything.

