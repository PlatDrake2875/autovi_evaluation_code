"""Tests for GaussianMechanism class."""

import math

import numpy as np
import pytest

from src.privacy import GaussianMechanism


class TestGaussianMechanismInit:
    """Tests for GaussianMechanism initialization."""

    def test_valid_initialization(self):
        """Test valid parameter initialization."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        assert mech.epsilon == 1.0
        assert mech.delta == 1e-5
        assert mech.sensitivity == 1.0
        assert mech.sigma > 0

    def test_epsilon_too_low(self):
        """Test error when epsilon is below minimum."""
        with pytest.raises(ValueError, match="Epsilon must be in"):
            GaussianMechanism(epsilon=0.05, delta=1e-5, sensitivity=1.0)

    def test_epsilon_too_high(self):
        """Test error when epsilon exceeds maximum."""
        with pytest.raises(ValueError, match="Epsilon must be in"):
            GaussianMechanism(epsilon=15.0, delta=1e-5, sensitivity=1.0)

    def test_delta_invalid(self):
        """Test error when delta is out of range."""
        with pytest.raises(ValueError, match="Delta must be in"):
            GaussianMechanism(epsilon=1.0, delta=0, sensitivity=1.0)
        with pytest.raises(ValueError, match="Delta must be in"):
            GaussianMechanism(epsilon=1.0, delta=1.0, sensitivity=1.0)

    def test_sensitivity_invalid(self):
        """Test error when sensitivity is not positive."""
        with pytest.raises(ValueError, match="Sensitivity must be positive"):
            GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=-1.0)
        with pytest.raises(ValueError, match="Sensitivity must be positive"):
            GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=0)


class TestSigmaComputation:
    """Tests for sigma (noise scale) computation."""

    def test_sigma_formula_correctness(self):
        """Verify sigma matches the Gaussian mechanism formula."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        expected_sigma = 1.0 * math.sqrt(2 * math.log(1.25 / 1e-5)) / 1.0
        assert abs(mech.sigma - expected_sigma) < 1e-10

    def test_sigma_scales_with_sensitivity(self):
        """Verify sigma scales linearly with sensitivity."""
        mech1 = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        mech2 = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=2.0)
        assert abs(mech2.sigma - 2 * mech1.sigma) < 1e-10

    def test_sigma_inverse_to_epsilon(self):
        """Verify sigma is inversely proportional to epsilon."""
        mech1 = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        mech2 = GaussianMechanism(epsilon=2.0, delta=1e-5, sensitivity=1.0)
        assert abs(mech1.sigma - 2 * mech2.sigma) < 1e-10

    def test_sigma_values_for_known_parameters(self):
        """Test sigma values for parameters from Stage2 plan."""
        # Expected values from Stage2_DP_Implementation_Plan.md
        mech_eps1 = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        mech_eps5 = GaussianMechanism(epsilon=5.0, delta=1e-5, sensitivity=1.0)
        mech_eps10 = GaussianMechanism(epsilon=10.0, delta=1e-5, sensitivity=1.0)

        # Approximate expected values: 4.83, 0.97, 0.48
        assert 4.8 < mech_eps1.sigma < 4.9
        assert 0.95 < mech_eps5.sigma < 1.0
        assert 0.45 < mech_eps10.sigma < 0.5


class TestNoiseAddition:
    """Tests for noise addition functionality."""

    def test_noise_shape_preserved(self):
        """Verify output shape matches input shape."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        data = np.random.randn(100, 50)
        noised = mech.add_noise(data, seed=42)
        assert noised.shape == data.shape

    def test_noise_dtype_preserved(self):
        """Verify output dtype matches input dtype."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        data = np.random.randn(100, 50).astype(np.float32)
        noised = mech.add_noise(data, seed=42)
        assert noised.dtype == data.dtype

    def test_noise_changes_data(self):
        """Verify noise actually modifies the data."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        data = np.random.randn(100, 50)
        noised = mech.add_noise(data, seed=42)
        assert not np.allclose(data, noised)

    def test_noise_variance_matches_sigma(self):
        """Verify added noise has the correct variance."""
        mech = GaussianMechanism(epsilon=5.0, delta=1e-5, sensitivity=1.0)
        data = np.zeros((10000, 100))
        noised = mech.add_noise(data, seed=42)
        measured_std = np.std(noised)
        # Allow 5% tolerance for statistical variation
        assert abs(measured_std - mech.sigma) / mech.sigma < 0.05

    def test_seed_reproducibility(self):
        """Verify same seed produces identical noise."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        data = np.random.randn(100, 50)
        noised1 = mech.add_noise(data, seed=123)
        noised2 = mech.add_noise(data, seed=123)
        np.testing.assert_array_equal(noised1, noised2)

    def test_different_seeds_different_noise(self):
        """Verify different seeds produce different noise."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        data = np.random.randn(100, 50)
        noised1 = mech.add_noise(data, seed=123)
        noised2 = mech.add_noise(data, seed=456)
        assert not np.allclose(noised1, noised2)

    def test_no_seed_varies(self):
        """Verify no seed produces different results each time."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        data = np.random.randn(100, 50)
        noised1 = mech.add_noise(data, seed=None)
        noised2 = mech.add_noise(data, seed=None)
        # Note: This could theoretically fail with astronomically low probability
        assert not np.allclose(noised1, noised2)


class TestRepr:
    """Tests for string representation."""

    def test_repr_format(self):
        """Test repr contains expected information."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        repr_str = repr(mech)
        assert "epsilon=1.0" in repr_str
        assert "delta=1e-05" in repr_str
        assert "sensitivity=1.0" in repr_str
        assert "sigma=" in repr_str
