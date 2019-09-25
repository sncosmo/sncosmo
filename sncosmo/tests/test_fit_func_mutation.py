#!/usr/bin/env python3.7
# -*- coding: UTF-8 -*-

"""Tests for the ``fit_funcs`` module."""

from copy import deepcopy
from unittest import TestCase

import sncosmo


class BaseTestingClass(TestCase):
    """Tests for an arbitrary sncosmo model"""

    def _test_mutation(self):
        """Test a pipeline fitting function does not mutate arguments"""

        # Use sncosmo example data for testing
        data = sncosmo.load_example_data()
        model = sncosmo.Model('salt2')
        params = model.param_names
        bounds = {'z': (0.3, 0.7)}

        # Preserve original input data
        original_data = deepcopy(data)
        original_model = deepcopy(model)
        original_bounds = deepcopy(bounds)
        original_params = deepcopy(params)

        # Check for argument mutation
        self.fit_func(data, model, vparam_names=model.param_names,
                      bounds=bounds)

        self.assertTrue(
            all(original_data == data),
            '``data`` argument was mutated')

        self.assertSequenceEqual(
            original_params, params,
            '``vparam_names`` argument was mutated')

        self.assertEqual(
            original_bounds, bounds,
            '``bounds`` argument was mutated')

        self.assertSequenceEqual(
            original_model.parameters.tolist(),
            model.parameters.tolist(),
            '``model`` argument was mutated')


class SimpleFit(BaseTestingClass):
    """Tests for the ``simple_fit`` function"""

    @staticmethod
    def fit_func(*a, **kw):
        return sncosmo.fit_lc(*a, **kw)

    def test_mutation(self):
        """Test arguments are not mutated"""

        self._test_mutation()


class NestFit(BaseTestingClass):
    """Tests for the ``nest_fit`` function"""

    @staticmethod
    def fit_func(*a, **kw):
        return sncosmo.nest_lc(*a, **kw)

    def test_mutation(self):
        """Test arguments are not mutated"""

        self._test_mutation()


class MCMCFit(BaseTestingClass):
    """Tests for the ``mcmc_fit`` function"""

    @staticmethod
    def fit_func(*a, **kw):
        return sncosmo.mcmc_lc(*a, **kw)

    def test_mutation(self):
        """Test arguments are not mutated"""

        self._test_mutation()
