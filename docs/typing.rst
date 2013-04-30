******************
Photometric Typing
******************

Example
-------

    >>> import numpy as np
    >>> from sncosmo.typing import ModelGrid
    ... 
    >>> # define grid of parameters
    >>> sn1a_params = {'z': np.logspace(-2, 0.07918, 100),
    ...                'x1': np.linspace(-4., 3., 10),
    ...                'c': np.linspace(-1., 1., 21),
    ...                'mabs': np.linspace(-20., -18., 21)}
    >>> sncc_params = {'z': np.logspace(-2, 0.07918, 100),
    ...                'c': np.linspace(-1., 1., 21),
    ...                'mabs': np.linspace(-19., -17., 21)}
    ... 
    >>> mg = ModelGrid()
    >>> mg.add_model('salt2', 'SN Ia', paramvals, prior=0.5)
    >>> mg.add_model('s11-2004hx', 'SN CC', paramvals, prior=0.25)
    >>> mg.add_model('s11-2005lc', 'SN CC', paramvals, prior=0.25)
    >>> mg.precompute(verbose=True)
    >>> p = mg.p(data)

Tying parameters
----------------

    >>> sn1a_params = {'z': np.logspace(-2, 0.07918, 100),
    ...                'x1': np.linspace(-4., 3., 10),
    ...                'c': np.linspace(-1., 1., 21),
    ...                'dmu': np.linspace(-1., 1., 21)}
    >>> def mabs(p):
    ...     return -19.3 - 0.12 * p['x1'] + 2.5 * p['c'] + p['dmu']
