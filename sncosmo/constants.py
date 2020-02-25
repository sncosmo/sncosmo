# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Constants used elsewhere in sncosmo."""

import astropy.constants as const
import astropy.units as u

BANDPASS_TRIM_LEVEL = 0.001
SPECTRUM_BANDFLUX_SPACING = 1.0
MODEL_BANDFLUX_SPACING = 5.0

H_ERG_S = const.h.cgs.value
C_AA_PER_S = const.c.to(u.AA / u.s).value
HC_ERG_AA = H_ERG_S * C_AA_PER_S
