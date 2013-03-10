*****************
Magnitude Systems
*****************


Local Magnitude Systems
-----------------------

Make the magnitude system a "local magnitude system" by setting offsets.

    >>> sdss = sncosmo.ABMagSystem(refmags={'sdssg': 0.02, 'sdssr':0.03})

"SDSS" magnitudes now differ from the AB system in that an object of
magnitude 0 in the AB system will have magnitude 0.02 the "SDSS"
magnitude system (for the 'sdssg' band). That is,
`sdssg(SDSS) = sdssg(AB) + 0.02`

You can also set the reference magnitudes after the fact:

    >>> sdss = sncosmo.MagSystem.from_name('ab')  # The AB system
    >>> sdss.refmags
    None
    >>> sdss.refmags = {'sdssg': 0.02, 'sdssr':0.03}

But, you can't assign to refmags once created.

    >>> sdss.refmags['sdssi'] = 0.04

This is because on access, sdss.refmags returns a copy.
