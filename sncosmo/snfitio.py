# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Functions for reading formats specific to snfit (SALT2) software.
(except lightcurves, which are in io.py).
"""

from collections import OrderedDict
import os

import numpy as np
from .io import _read_salt2
from .bandpasses import Bandpass, BandpassInterpolator


def _collect_scalars(values):
    """Given a list containing scalars (float or int) collect scalars
    into a single prefactor. Input list is modified."""

    prefactor = 1.0
    for i in range(len(values)-1, -1, -1):
        if isinstance(values[i], (int, float)):
            prefactor *= values.pop(i)

    return prefactor


def _parse_value(s):
    try:
        x = int(s)
    except:
        try:
            x = float(s)
        except:
            x = s
    return x


def read_cards(fname):
    cards = OrderedDict()
    with open(fname, 'r') as f:
        for line in f:
            if line[0] != '@':
                continue

            words = line.split()
            key = words[0][1:]  # words is at least length 1
            tokens = words[1:]
            if len(tokens) == 0:
                value = None
            elif len(tokens) == 1:
                value = _parse_value(tokens[0])
            else:
                value = [_parse_value(v) for v in tokens]

            cards[key] = value

    return cards


def read_filterwheel(fname):
    """Read a single definition from a filterwheel file."""

    # find filter in filterwheel file
    result = {}
    with open(fname, 'r') as f:
        for line in f:
            words = line.split()  # each line can have 2 or 3 words.
            if len(words) not in (2, 3):
                raise Exception("illegal line in filterwheel file")
            result[words[0]] = words[-1]

    return result


def read_radial_filterwheel(fname):
    """Read radially variable filterwheel transmissions.

    Parameters
    ----------
    fname : str


    Returns
    -------
    dict of list
        Dictionary where keys are filter names and values are lists. Each
        item in the list is a two-tuple giving the radius and filter function
        (in the form of a SampledFunction).
    """

    # read filter filenames (multiple files per filter)
    result = {}
    with open(fname, 'r') as f:
        for line in f:
            band, _, bandfname = line.split()
            if band not in result:
                result[band] = []
            result[band].append(bandfname)

    return result


def read_snfit_bandpass(dirname, filtername):
    """Read an snfit-format bandpass.

    The instrument.cards file should contain a ``FILTERS`` keyword.

    Parameters
    ----------
    dirname : str
        Directory in which to look for files. Should contain an
        ``instrument.cards`` file.
    filtername : str
        Filter name (typically a single lowercase letter).

    Returns
    -------
    Bandpass, AggregateBandpass
    """

    # we do this a lot below
    def expand(x):
        return os.path.join(dirname, x)

    cards = read_cards(expand("instrument.cards"))

    # Check that this is really a bandpass and not a generator.
    if "FILTERS" not in cards:
        raise Exception("FILTERS key not in instrument.cards file. For "
                        "radially variable filters (with "
                        "RADIALLY_VARIABLE_FILTER KEY) use "
                        "`read_snfit_bandpass_generator`.")

    transmissions = []  # scalars or (wave, trans) pairs

    # required keys (transmissions apply to all filters)
    for key in ("MIRROR_REFLECTIVITY", "OPTICS_TRANS", "QE",
                "ATMOSPHERIC_TRANS"):
        value = cards[key]
        if type(value) is int or type(value) is float:
            transmissions.append(value)
        else:
            x, y = np.loadtxt(expand(value), unpack=True)
            transmissions.append((x, y))

    # optional key (filter-specific)
    if "CHROMATIC_CORRECTIONS" in cards:
        fname = read_filterwheel(
            expand(cards["CHROMATIC_CORRECTIONS"]))[filtername]
        x, y = np.loadtxt(expand(fname), unpack=True, skiprows=3)
        transmissions.append((x, y))

    # Read filters
    fname = read_filterwheel(expand(cards["FILTERS"]))[filtername]
    x, y = np.loadtxt(expand(fname), unpack=True)
    transmissions.append((x, y))

    # simplify transmissions
    prefactor = _collect_scalars(transmissions)

    # if there's only one non-scalar, we can construct a normal
    # bandpass
    if len(transmissions) == 1:
        wave, trans = transmissions[0]
        return sncosmo.Bandpass(wave, prefactor * trans, wave_unit=u.AA)
    elif len(transmissions) > 1:
        return AggregateBandpass(transmissions, prefactor=prefactor)
    else:
        raise Exception("bandpass consists only of scalars")


def read_snfit_bandpass_interpolator(dirname, filtername, name=None):
    """Read an snfit-format bandpass or bandpass generator.

    Parameters
    ----------
    dirname : str
        Directory in which to look for files. Should contain an
        ``instrument.cards`` file.
    filtername : str
        Filter name (typically a single lowercase letter).

    Returns
    -------
    BandpassInterpolator
    """

    # we do this a lot below
    def expand(x):
        return os.path.join(dirname, x)

    cards = read_cards(expand("instrument.cards"))

    transmissions = []  # scalars or (wave, trans) pairs

    # Check that this is really a generator.
    if "RADIALLY_VARIABLE_FILTERS" not in cards:
        raise Exception("RADIALLY_VARIABLE_FILTERS key not in "
                        "instrument.cards file. For non-variable "
                        "filters (with FILTER key) use `read_snfit_bandpass`.")

    # required keys (transmissions apply to all filters)
    for key in ("MIRROR_REFLECTIVITY", "OPTICS_TRANS", "QE",
                "ATMOSPHERIC_TRANS"):
        value = cards[key]
        if type(value) is int or type(value) is float:
            transmissions.append(value)
        else:
            x, y = np.loadtxt(expand(value), unpack=True)
            transmissions.append((x, y))

    # optional key:
    if "CHROMATIC_CORRECTIONS" in cards:
        corr_fnames = read_filterwheel(expand(cards["CHROMATIC_CORRECTIONS"]))

        # skip if correction file not defined for this band
        if filtername in corr_fnames:
            fname = corr_fnames[filtername]
            x, y = np.loadtxt(expand(fname), unpack=True, skiprows=3)
            transmissions.append((x, y))

    fnames = read_radial_filterwheel(
        expand(cards["RADIALLY_VARIABLE_FILTERS"]))[filtername]

    # read transmission functions at each radius
    radial_transmissions = []
    for fname in fnames:
        # TODO: re-organize the salt2-format reader.
        with open(expand(fname), 'r') as f:
            meta, data = _read_salt2(f)

        try:
            r_str = meta["MEASUREMENT_RADIUS"]
        except KeyError:
            raise Exception("MEASUREMENT_RADIUS keyword not found in " +
                            os.path.join(dirname, bandfname))

        r = float(r_str.split()[0])  # parse string like '0 cm'
        radial_transmissions.append((r, data['lambda'], data['tr']))

    # simplify the list
    prefactor = _collect_scalars(transmissions)

    return BandpassInterpolator(transmissions, radial_transmissions,
                                prefactor=prefactor, name=name)
