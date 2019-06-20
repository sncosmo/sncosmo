#!/bin/bash -x

hash -r
conda config --add channels conda-forge
conda config --set always_yes yes --set changeps1 no
conda update -q --yes conda
conda info -a

# EGG_INFO
if [[ $MAIN_CMD == *egg_info* ]]
then
  export CONDA_PACKAGES="python=$PYTHON_VERSION pip"
# STYLE
elif [[ $MAIN_CMD == *checkstyle* ]]
then
  export CONDA_PACKAGES="python=$PYTHON_VERSION pip pycodestyle"
# EVERYTHING ELSE:
else
  # core dependencies
  export CONDA_PACKAGES="python=$PYTHON_VERSION pip pytest  numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION astropy=$ASTROPY_VERSION cython extinction"

  # optional dependencies
  if $OPTIONAL_DEPS
  then
    export CONDA_PACKAGES="$CONDA_PACKAGES matplotlib iminuit"
    export PIP_INSTALL="pip install --no-deps nestle"
  fi

  # doc dependencies
  if [[ $MAIN_CMD == *docs* ]]
  then
    export CONDA_PACKAGES="$CONDA_PACKAGES sphinx sphinx-gallery pygments matplotlib pillow sphinx_rtd_theme numpydoc"
  fi

  # coverage dependencies
  if [[ $MAIN_CMD == *"--cov"* ]]
  then
    export CONDA_PACKAGES="$CONDA_PACKAGES coverage coveralls pytest-cov"
  fi
fi

conda create --yes -n test $CONDA_PACKAGES
source activate test
$PIP_INSTALL