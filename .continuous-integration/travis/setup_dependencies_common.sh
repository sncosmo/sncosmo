#!/bin/bash -x

hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a

# CONDA
conda create --yes -n test -c astropy-ci-extras python=$PYTHON_VERSION pip
source activate test

# --no-use-wheel requirement is temporary due to
# https://github.com/astropy/astropy/issues/4180
# and may be removed once the upstream fix is in place
export PIP_INSTALL="pip install --no-deps --no-use-wheel"

# Now set up shortcut to conda install command to make sure the Python and Numpy
# versions are always explicitly specified.
export CONDA_INSTALL="conda install --yes python=$PYTHON_VERSION numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION"

# EGG_INFO
if [[ $SETUP_CMD == egg_info ]]
then
  return  # no more dependencies needed
fi

# PEP8
if [[ $MAIN_CMD == pep8* ]]
then
  $PIP_INSTALL pep8
  return  # no more dependencies needed
fi

# CORE DEPENDENCIES (besides astropy)
$CONDA_INSTALL pip jinja2 psutil
$PIP_INSTALL extinction

# ASTROPY
if [[ $ASTROPY_VERSION == dev ]]
then
  $PIP_INSTALL git+http://github.com/astropy/astropy.git
else
  $CONDA_INSTALL astropy=$ASTROPY_VERSION
fi

# OPTIONAL DEPENDENCIES
if $OPTIONAL_DEPS
then
  $CONDA_INSTALL matplotlib
  $PIP_INSTALL emcee nestle iminuit
fi

# DOCUMENTATION DEPENDENCIES
# build_sphinx needs sphinx as well as matplotlib (for plot_directive)
# Note that this matplotlib will *not* work with py 3.x, but our sphinx
# build is currently 2.7, so that's fine
if [[ $SETUP_CMD == build_sphinx* ]]
then
  $PIP_INSTALL sphinx-gallery astropy-helpers
  $CONDA_INSTALL sphinx pygments matplotlib pillow sphinx_rtd_theme
fi

# COVERAGE DEPENDENCIES
if [[ $SETUP_CMD == 'test --coverage' ]]
then
  # TODO can use latest version of coverage (4.0) once
  # https://github.com/astropy/astropy/issues/4175 is addressed in
  # astropy release version.
  pip install coverage==3.7.1
  pip install coveralls
fi
