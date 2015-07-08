
Simulation
==========

First, define a set of "observations". These are the properties of our
observations: the time, bandpass and depth.

.. code:: python

    import sncosmo
    from astropy.table import Table
    obs = Table({'time': [56176.19, 56188.254, 56207.172],
                 'band': ['desg', 'desr', 'desi'],
                 'gain': [1., 1., 1.],
                 'skynoise': [191.27, 147.62, 160.40],
                 'zp': [30., 30., 30.],
                 'zpsys':['ab', 'ab', 'ab']})
    print obs

.. parsed-literal::

    skynoise zpsys band gain    time    zp 
    -------- ----- ---- ---- --------- ----
      191.27    ab desg  1.0  56176.19 30.0
      147.62    ab desr  1.0 56188.254 30.0
       160.4    ab desi  1.0 56207.172 30.0


Suppose we want to simulate a SN with the SALT2 model and the following
parameters:

.. code:: python

    model = sncosmo.Model(source='salt2')
    params = {'z': 0.4, 't0': 56200.0, 'x0':1.e-5, 'x1': 0.1, 'c': -0.1}

To get the light curve for this single SN, we'd do:

.. code:: python

    lcs = sncosmo.realize_lcs(obs, model, [params])
    print lcs[0]

.. parsed-literal::

       time   band      flux        fluxerr     zp  zpsys
    --------- ---- ------------- ------------- ---- -----
     56176.19 desg 96.0531272705  191.27537908 30.0    ab
    56188.254 desr 456.360196623  149.22627064 30.0    ab
    56207.172 desi  655.40885611 162.579572369 30.0    ab


Note that we've passed the function a one-element list, ``[params]``,
and gotten back a one-element list in return. (The ``realize_lcs``
function is designed to operate on lists of SNe for convenience.)

Generating SN parameters
------------------------

We see above that it is straightforward to simulate SNe once we already
know the parameters of each one. But what if we want to pick SN
parameters from some defined distribution?

Suppose we want to generate SN parameters for all the SNe we would find
in a given search area over a defined period of time. We start by
defining an area and time period, as well as a maximum redshift to
consider:

.. code:: python

    area = 1.  # area in square degrees
    tmin = 56175.  # minimum time
    tmax = 56225.  # maximum time
    zmax = 0.7

First, we'd like to get the number and redshifts of all SNe that occur
over our 1 square degree and 50 day time period:

.. code:: python

    redshifts = list(sncosmo.zdist(0., zmax, time=(tmax-tmin), area=area))
    print len(redshifts), "SNe"
    print "redshifts:", redshifts

.. parsed-literal::

    9 SNe
    redshifts: [0.4199710008856507, 0.3500118339133868, 0.5915676316485601, 0.5857452631151785, 0.49024466410556855, 0.5732679644841575, 0.6224436826380927, 0.5853477892182203, 0.5522300320124105]


Generate a list of SN parameters using these redshifts, drawing ``x1``
and ``c`` from normal distributions:

.. code:: python

    from numpy.random import uniform, normal
    params = [{'x0':1.e-5, 'x1':normal(0., 1.), 'c':normal(0., 0.1),
               't0':uniform(tmin, tmax), 'z': z}
              for z in redshifts]
    for p in params:
        print p

.. parsed-literal::

    {'z': 0.4199710008856507, 'x0': 1e-05, 'x1': -0.9739877070754421, 'c': -0.1465835504611458, 't0': 56191.57686616353}
    {'z': 0.3500118339133868, 'x0': 1e-05, 'x1': 0.04454878604727126, 'c': -0.04920811869083081, 't0': 56222.76963606611}
    {'z': 0.5915676316485601, 'x0': 1e-05, 'x1': -0.26765265677262423, 'c': -0.06456008680932701, 't0': 56211.706219411404}
    {'z': 0.5857452631151785, 'x0': 1e-05, 'x1': 0.8255953341731204, 'c': 0.08520083775049729, 't0': 56209.33583211229}
    {'z': 0.49024466410556855, 'x0': 1e-05, 'x1': -0.12051827966517584, 'c': -0.09490756669333822, 't0': 56189.37571007927}
    {'z': 0.5732679644841575, 'x0': 1e-05, 'x1': 0.3051310078192594, 'c': -0.10967604820261241, 't0': 56198.04368422346}
    {'z': 0.6224436826380927, 'x0': 1e-05, 'x1': -0.6329407028587257, 'c': -0.009789183239376284, 't0': 56179.88133113836}
    {'z': 0.5853477892182203, 'x0': 1e-05, 'x1': 0.6373371286596669, 'c': 0.05151693090038232, 't0': 56212.04579735962}
    {'z': 0.5522300320124105, 'x0': 1e-05, 'x1': 0.04762095339856289, 'c': -0.005018877828783951, 't0': 56182.14827040906}


So far so good. The only problem is that ``x0`` doesn't vary. We'd like
it to be randomly distributed with some scatter around the Hubble line,
so it should depend on the redshift. Here's an alternative:

.. code:: python

    params = []
    for z in redshifts:
        mabs = normal(-19.3, 0.3)
        model.set(z=z)
        model.set_source_peakabsmag(mabs, 'bessellb', 'ab')
        x0 = model.get('x0')
        p = {'z':z, 't0':uniform(tmin, tmax), 'x0':x0, 'x1': normal(0., 1.), 'c': normal(0., 0.1)}
        params.append(p)
    
    for p in params:
        print p


.. parsed-literal::

    {'c': -0.060104568346581566, 'x0': 2.9920355958896461e-05, 'z': 0.4199710008856507, 'x1': -0.677121283126299, 't0': 56217.93979718883}
    {'c': 0.10405991801014292, 'x0': 2.134500759148091e-05, 'z': 0.3500118339133868, 'x1': 1.6034252041294512, 't0': 56218.008314206476}
    {'c': -0.14777109151711296, 'x0': 7.9108889725043354e-06, 'z': 0.5915676316485601, 'x1': -2.2082282760850993, 't0': 56218.013686428785}
    {'c': 0.056034777154805086, 'x0': 6.6457371815973038e-06, 'z': 0.5857452631151785, 'x1': 0.675413080007434, 't0': 56189.03517395757}
    {'c': -0.0709158052635228, 'x0': 1.2228145655148946e-05, 'z': 0.49024466410556855, 'x1': 0.5449847454420981, 't0': 56198.02895700289}
    {'c': -0.22101146234021096, 'x0': 7.4299221264917702e-06, 'z': 0.5732679644841575, 'x1': -1.543245858395605, 't0': 56189.04585414441}
    {'c': 0.06964843664572477, 'x0': 9.7121906557832662e-06, 'z': 0.6224436826380927, 'x1': 1.7419604610283943, 't0': 56212.827270197355}
    {'c': 0.07320513053870191, 'x0': 3.22205341646521e-06, 'z': 0.5853477892182203, 'x1': -0.39658066375434153, 't0': 56200.421464066916}
    {'c': 0.18555773972769227, 'x0': 7.5955258508017471e-06, 'z': 0.5522300320124105, 'x1': -0.24463691193386283, 't0': 56190.492271332616}


Now we can generate the lightcurves for these parameters:

.. code:: python

    lcs = sncosmo.realize_lcs(obs, model, params)
    print lcs[0]

.. parsed-literal::

       time   band      flux       fluxerr     zp  zpsys
    --------- ---- ------------- ------------ ---- -----
     56176.19 desg 6.70520005464       191.27 30.0    ab
    56188.254 desr 106.739113709       147.62 30.0    ab
    56207.172 desi  1489.7521011 164.62420476 30.0    ab


Note that the "true" parameters are saved in the metadata of each SN:

.. code:: python

    lcs[0].meta



.. parsed-literal::

    {'c': -0.060104568346581566,
     't0': 56217.93979718883,
     'x0': 2.9920355958896461e-05,
     'x1': -0.677121283126299,
     'z': 0.4199710008856507}



