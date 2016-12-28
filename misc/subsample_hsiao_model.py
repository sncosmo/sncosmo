#!/usr/bin/env python
"""
Usage: %prog INFILE OUTFILE

Subsample the Hsiao spectral time series, for use as a small demonstration
model that can be included with source code."""

from optparse import OptionParser
import numpy as np
from sncosmo.io import read_griddata_fits, write_griddata_fits

parser = OptionParser()
options, args = parser.parse_args()

phase, wave, flux = read_griddata_fits(args[0])
write_griddata_fits(phase[::5], wave[::5], flux[::5, ::5], args[1])
print "wrote to", args[1]
