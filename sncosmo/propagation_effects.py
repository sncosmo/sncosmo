# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""PropagationEffect classes - models of the effect on spectral flux density
of a source."""

from .extinction import extinction
from textwrap import dedent

class PropagationEffect(object):
    """Base class for propagation effects."""
    
    __metaclass__ = abc.ABCMeta

    @property
    def parameters(self):
        return self._parameters[:]

    @parameters.setter
    def parameters(self, value):
        self._parameters[:] = value

    @abstractmethod
    def propagate(self, wave, flux):
        pass

class Dust(PropagationEffect):
    """Dust propagation effect. Wraps 

    Parameters
    ----------
    model : str, optional
        The dust extinction model to use.

    See Also
    --------
    extinction
    """

    param_names = ['ebv']

    def __init__(self, model='f99', r_v=3.1, minwave=1000., maxwave=30000.,
                 spline_points=2000):
        self._parameters = np.array([0.])
        self._minwave = minwave
        self._maxwave = maxwave
        self._wave = np.logspace(np.log10(minwave), np.log10(maxwave),
                                 spline_points)
        self._model = model
        self.r_v = r_v

    @property
    def r_v(self):
        """R_V value"""
        return self._r_v

    @r_v.setter
    def r_v(self, value):
        self._r_v = value

        # Recalculate spline
        trans_base = 10.**(-0.4 * extinction(self._wavelengths, ebv=1.,
                                             r_v=value, model=self._model))
        self._spline = Spline1d(self._wave, trans_base)

    def propagate(self, wave, flux):
        """Propagate the flux."""
        return flux * self._spline(wave) ** self._parameters[0]

    def __str__(self):
        result = """\
        {}(model={}, r_v={}, minwave={}, maxwave={}, spline_points={})
        Parameters:""".format(self.__class__.__name__, self._model,
                              self._wave[0], self._wave[-1], len(self._wave))
        return dedent(result)
