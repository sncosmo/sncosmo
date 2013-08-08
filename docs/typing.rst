******************
Light Curve Typing
******************

Photometric typing in `sncosmo` is carried out by computing the
Baysian evidence for each model in a set of models given the data,
then comparing the evidence for each model. The evidence for each
model depends necessarily on the priors for each parameter in the
model.

The class `~sncosmo.PhotoTyper` allows the user to fully configure
which models are used and the priors for each parameter in each model.

Simple Example
==============

First, initialize a new typer. To start, it has no models:

    >>> typer = sncosmo.PhotoTyper()  

Add a model for Type Ia SNe, and supply a range of model
parameters. By default, the prior for each parameter will be flat
inside this range and zero outside.

    >>> sn1a_parlims = {'z': (0.01, 1.2), 'c':(-0.4, 0.6), 's': (0.7, 1.3),
    ...                 'mabs':(-18., -20.)}
    >>> typer.add_model('hsiao', 'SN Ia', sn1a_parlims)

Add two models for Type II SNe:

    >>> sncc_parlims = {'z': (0.01, 1.1), 'c':(0., 0.6), 'mabs':(-17., -19.)}
    >>> typer.add_model('s11-2004hx', 'SN IIL', sncc_parlims)
    >>> typer.add_model('s11-2005lc', 'SN IIP', sncc_parlims)

Read light curve data and classify it:

    >>> meta, data = sncosmo.readlc(args.filename)
    >>> types, models = typer.classify(data)


Tying parameters
================

    >>> sn1a_parlims = {'z': (0.01, 1.2), 'c':(-0.4, 0.6), 's': (0.7, 1.3),
    ...                 'dmu':(-1., 1.)}
    >>> def mabs(p):
    ...     return -19.3 - 1.3 * (p['s'] - 1.) + 2.5 * p['c'] + p['dmu']
    >>> typer.add_model('hsiao', 'SN Ia', sn1a_parlims, tied={'mabs': mabs})
