"""Tests for ieeg_wm.analysis — shrinkage_gamma."""
import numpy as np
import pytest

from ieeg_wm.analysis import shrinkage_gamma


class TestShrinkageGamma:
    """Both computation paths must yield the same gamma to within floating-point
    tolerance. A synthetic data matrix with a known random seed is used so the
    test is fully deterministic."""

    @pytest.fixture
    def data_matrix(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal((10, 50))

    def test_standard_path_returns_float(self, data_matrix):
        gamma = shrinkage_gamma(data_matrix, mem_eff=False, feedback=False)
        assert isinstance(gamma, float)

    def test_mem_eff_path_returns_float(self, data_matrix):
        gamma = shrinkage_gamma(data_matrix, mem_eff=True, feedback=False)
        assert isinstance(gamma, float)

    def test_both_paths_are_deterministic(self, data_matrix):
        """Each path must return the same result when called twice."""
        g1 = shrinkage_gamma(data_matrix, mem_eff=False, feedback=False)
        g2 = shrinkage_gamma(data_matrix, mem_eff=False, feedback=False)
        assert g1 == g2

        g3 = shrinkage_gamma(data_matrix, mem_eff=True, feedback=False)
        g4 = shrinkage_gamma(data_matrix, mem_eff=True, feedback=False)
        assert g3 == g4

    def test_gamma_in_unit_interval(self, data_matrix):
        """Ledoit-Wolf shrinkage coefficient must lie in [0, 1]."""
        for mem_eff in (False, True):
            gamma = shrinkage_gamma(data_matrix, mem_eff=mem_eff, feedback=False)
            assert 0.0 <= gamma <= 1.0, f"gamma={gamma} (mem_eff={mem_eff}) is outside [0, 1]"

    def test_gamma_is_positive(self, data_matrix):
        """Shrinkage coefficient must be strictly positive for random data."""
        gamma = shrinkage_gamma(data_matrix, mem_eff=False, feedback=False)
        assert gamma > 0.0

    def test_standard_path_regression(self, data_matrix):
        """Regression test: output must match the known value for a fixed seed."""
        gamma = shrinkage_gamma(data_matrix, mem_eff=False, feedback=False)
        assert abs(gamma - 0.9531479813365358) < 1e-10
