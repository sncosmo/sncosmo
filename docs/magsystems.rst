*****************
Magnitude Systems
*****************

A `sncosmo.MagSystem` object represents an astrophysical magnitude
system and can be used to convert physical fluxes to magnitudes (in
that system).  For example, this gives the flux (in photons/s/cm^2) of
a source with AB magnitude zero in the Bessell B bandpass::

    >>> abmagsys = sncosmo.get_magsystem('ab')
    >>> abmagsys.zpbandflux('bessellb')
    1199275.6325987775
