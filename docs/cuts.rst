*************
Applying cuts
*************

It is useful to be able to apply "cuts" to data before trying to fit a
model to the data. This is particularly important when using some of
the "guessing" algorithms in `~sncosmo.fit_lc` and `~sncosmo.nest_lc`
that use a minimum signal-to-noise ratio to pick "good" data
points. These algorithms will raise an exception if there are no data
points meeting the requirements, so it is advisable to check if the
data meets the requirements beforehand.

Signal-to-noise ratio cuts
==========================

Require at least one datapoint with signal-to-noise ratio (S/N) greater than 5
(in any band):

    >>> pass = np.max(data['flux'] / data['fluxerr']) > 5.
    >>> pass
    True

Require two bands each with at least one datapoint having S/N > 5:

    >>> mask = data['flux'] / data['fluxerr'] > 5.
    >>> pass = len(np.unique(data['band'][mask])) >= 2
    >>> pass
    True
