========
Examples
========

Create a model
--------------

    >>> import sncosmo
    >>> model = sncosmo.get_model('salt2')
    >>> model
    <SALT2Model 'salt2' version='2.0' at 0x25fbb50>

Print a summary of the model
----------------------------

    >>> print model
    Model class: SALT2Model
    Model name: salt2
    Model version: 2.0
    Model phases: [-20, .., 50] days (71 points)
    Model dispersion: [2000, .., 9200] Angstroms (721 points) 
    Reference phase: -0.656900003922 days
    Cosmology: WMAP9(H0=69.3, Om0=0.286, Ode0=0.713)
               (lum. distance = None)
    Current Parameters:
        fscale = 1.0
        m = None [bessellb, ab]
        mabs = None [bessellb, ab]
        t0 = 0.0
        z = None
        c = 0.0
        x1 = 0.0

Set model parameters
--------------------

    >>> model.set(c=0.05, x1=0.5, mabs=-19.3, z=1.0, t0=56000.)

Get a spectrum
--------------

    >>> model.flux(56010.)
    array([  3.38760601e-31,   3.40124899e-23,   1.36596249e-22, ...

Get synthetic photometry
------------------------

In photons / s / cm^2:

    >>> model.bandflux('sdssr', [56010., 56020., 56030., 56040., 56050.])
    array([  1.48745060e-05,   9.14207795e-06,   4.65357185e-06,
             2.40957277e-06,   1.20706548e-06])

Scaled to a given zeropoint (and zeropoint magnitude system):

    >>> model.bandflux('sdssr', [56010., 56020., 56030., 56040., 56050.],
    ...                zp=25., zpmagsys='ab')
    array([ 0.30141438,  0.18525346,  0.09429916,  0.04882716,  0.02445976])


For more detailed usage, see :doc:`models`.

Many built in models such as the Hsiao, Nugent, PSNID, and SALT2 models.
See the :ref:`list-of-built-in-models`.

