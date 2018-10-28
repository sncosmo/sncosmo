#!/bin/bash -x

hash -r
conda config --add channels conda-forge
conda config --set always_yes yes --set changeps1 no
conda update -q --yes conda
conda info -a

# CONDA
conda create --yes -n test python=$PYTHON_VERSION pip
source activate test

# --no-use-wheel requirement is temporary due to
# https://github.com/astropy/astropy/issues/4180
# and may be removed once the upstream fix is in place
export PIP_INSTALL="pip install --no-deps"

# Now set up shortcut to conda install command to make sure the Python and Numpy
# versions are always explicitly specified.
export CONDA_INSTALL="conda install --yes python=$PYTHON_VERSION numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION"

# EGG_INFO
if [[ $SETUP_CMD == egg_info ]]
then
  return  # no more dependencies needed
fi

# PEP8
if [[ $MAIN_CMD == *checkstyle* ]]
then
  conda install --yes python=$PYTHON_VERSION pycodestyle
  return  # no more dependencies needed
fi

# CORE DEPENDENCIES (besides astropy)
$CONDA_INSTALL jinja2 psutil cython extinction

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
  $CONDA_INSTALL matplotlib iminuit
  $PIP_INSTALL nestle
fi

# DOCUMENTATION DEPENDENCIES
if [[ $SETUP_CMD == *docs* ]]
then
  $CONDA_INSTALL sphinx sphinx-gallery pygments matplotlib pillow sphinx_rtd_theme numpydoc
fi

# COVERAGE DEPENDENCIES
if [[ $SETUP_CMD == *"--coverage"* ]]
then
  $CONDA_INSTALL coverage coveralls
fi
