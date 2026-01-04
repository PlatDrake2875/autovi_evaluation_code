"""Tests for fairness metric calculations."""

import pytest
import numpy as np

from src.fairness.metrics import (
    FairnessMetrics,
    compute_jains_index,
    compute_performance_variance,
    compute_performance_gap,
    compute_worst_case,
    compute_coefficient_of_variation,
    compute_all_metrics,
)


class TestJainsIndex:
    """Tests for Jain's Fairness Index computation."""

    def test_perfect_fairness(self):
        """Test that equal performances give Jain's Index = 1.0."""
        performances = [0.8, 0.8, 0.8, 0.8]
        result = compute_jains_index(performances)
        assert result == pytest.approx(1.0)

    def test_perfect_fairness_different_values(self):
        """Test that any equal values give Jain's Index = 1.0."""
        performances = [0.5, 0.5, 0.5]
        result = compute_jains_index(performances)
        assert result == pytest.approx(1.0)

    def test_worst_fairness_two_groups(self):
        """Test that one group having all gives Jain's Index = 1/n."""
        # One group has everything, others have nothing
        performances = [1.0, 0.0]
        result = compute_jains_index(performances)
        assert result == pytest.approx(0.5)  # 1/2

    def test_worst_fairness_four_groups(self):
        """Test worst case with 4 groups."""
        performances = [1.0, 0.0, 0.0, 0.0]
        result = compute_jains_index(performances)
        assert result == pytest.approx(0.25)  # 1/4

    def test_intermediate_fairness(self):
        """Test intermediate fairness values."""
        performances = [0.9, 0.8, 0.7, 0.6]
        result = compute_jains_index(performances)
        # Jain's index should be between 1/4 and 1.0
        assert 0.25 < result < 1.0
        # Manual calculation: (3.0)^2 / (4 * 2.30) = 9 / 9.2 ≈ 0.978
        expected = 9.0 / (4 * (0.81 + 0.64 + 0.49 + 0.36))
        assert result == pytest.approx(expected)

    def test_single_element(self):
        """Test with single element (should be 1.0)."""
        performances = [0.8]
        result = compute_jains_index(performances)
        assert result == pytest.approx(1.0)

    def test_empty_raises_error(self):
        """Test that empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_jains_index([])

    def test_all_zeros_raises_error(self):
        """Test that all zeros raises ValueError."""
        with pytest.raises(ValueError, match="all performances are zero"):
            compute_jains_index([0.0, 0.0, 0.0])

    def test_bounds(self):
        """Test that Jain's index is always in [1/n, 1]."""
        rng = np.random.default_rng(42)
        for n in [2, 3, 5, 10]:
            performances = rng.uniform(0.1, 1.0, size=n).tolist()
            result = compute_jains_index(performances)
            assert 1 / n <= result <= 1.0


class TestPerformanceVariance:
    """Tests for performance variance computation."""

    def test_equal_performances(self):
        """Test that equal performances have zero variance."""
        performances = [0.8, 0.8, 0.8]
        result = compute_performance_variance(performances)
        assert result == pytest.approx(0.0)

    def test_known_variance(self):
        """Test variance with known values."""
        performances = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_performance_variance(performances)
        # Mean = 3.0, Var = ((1-3)^2 + ... + (5-3)^2) / 5 = 10/5 = 2.0
        assert result == pytest.approx(2.0)

    def test_single_element(self):
        """Test variance with single element is zero."""
        performances = [0.8]
        result = compute_performance_variance(performances)
        assert result == pytest.approx(0.0)

    def test_empty_raises_error(self):
        """Test that empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_performance_variance([])


class TestPerformanceGap:
    """Tests for performance gap (max - min) computation."""

    def test_equal_performances(self):
        """Test that equal performances have zero gap."""
        performances = [0.8, 0.8, 0.8]
        result = compute_performance_gap(performances)
        assert result == pytest.approx(0.0)

    def test_known_gap(self):
        """Test gap with known values."""
        performances = [0.5, 0.7, 0.9]
        result = compute_performance_gap(performances)
        assert result == pytest.approx(0.4)

    def test_single_element(self):
        """Test gap with single element is zero."""
        performances = [0.8]
        result = compute_performance_gap(performances)
        assert result == pytest.approx(0.0)

    def test_empty_raises_error(self):
        """Test that empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_performance_gap([])


class TestWorstCase:
    """Tests for worst-case (minimum) performance computation."""

    def test_known_minimum(self):
        """Test minimum with known values."""
        performances = [0.9, 0.5, 0.7]
        result = compute_worst_case(performances)
        assert result == pytest.approx(0.5)

    def test_equal_performances(self):
        """Test minimum when all equal."""
        performances = [0.8, 0.8, 0.8]
        result = compute_worst_case(performances)
        assert result == pytest.approx(0.8)

    def test_single_element(self):
        """Test minimum with single element."""
        performances = [0.8]
        result = compute_worst_case(performances)
        assert result == pytest.approx(0.8)

    def test_empty_raises_error(self):
        """Test that empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_worst_case([])


class TestCoefficientOfVariation:
    """Tests for coefficient of variation (std/mean) computation."""

    def test_equal_performances(self):
        """Test that equal performances have zero CV."""
        performances = [0.8, 0.8, 0.8]
        result = compute_coefficient_of_variation(performances)
        assert result == pytest.approx(0.0)

    def test_known_cv(self):
        """Test CV with known values."""
        performances = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_coefficient_of_variation(performances)
        # Mean = 3.0, Std = sqrt(2.0) ≈ 1.414
        # CV = 1.414 / 3.0 ≈ 0.471
        expected = np.std(performances) / np.mean(performances)
        assert result == pytest.approx(expected)

    def test_all_zeros(self):
        """Test CV with all zeros returns 0."""
        performances = [0.0, 0.0, 0.0]
        result = compute_coefficient_of_variation(performances)
        assert result == pytest.approx(0.0)

    def test_single_element(self):
        """Test CV with single element is zero."""
        performances = [0.8]
        result = compute_coefficient_of_variation(performances)
        assert result == pytest.approx(0.0)

    def test_empty_raises_error(self):
        """Test that empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_coefficient_of_variation([])


class TestComputeAllMetrics:
    """Tests for computing all fairness metrics at once."""

    def test_returns_fairness_metrics(self):
        """Test that compute_all_metrics returns FairnessMetrics."""
        group_performances = {"A": 0.8, "B": 0.7, "C": 0.9}
        result = compute_all_metrics(group_performances)
        assert isinstance(result, FairnessMetrics)

    def test_all_fields_populated(self):
        """Test that all fields are populated."""
        group_performances = {"A": 0.8, "B": 0.7, "C": 0.9}
        result = compute_all_metrics(group_performances)

        assert result.jains_index > 0
        assert result.variance >= 0
        assert result.performance_gap >= 0
        assert result.worst_case == 0.7
        assert result.coefficient_of_variation >= 0
        assert result.mean == pytest.approx(0.8)
        assert result.n_groups == 3
        assert result.group_performances == group_performances

    def test_empty_raises_error(self):
        """Test that empty dict raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_all_metrics({})

    def test_perfect_fairness(self):
        """Test metrics for perfectly fair scenario."""
        group_performances = {"A": 0.8, "B": 0.8, "C": 0.8}
        result = compute_all_metrics(group_performances)

        assert result.jains_index == pytest.approx(1.0)
        assert result.variance == pytest.approx(0.0)
        assert result.performance_gap == pytest.approx(0.0)
        assert result.coefficient_of_variation == pytest.approx(0.0)

    def test_stores_group_performances(self):
        """Test that group performances are stored correctly."""
        group_performances = {"client_0": 0.85, "client_1": 0.72, "client_2": 0.91}
        result = compute_all_metrics(group_performances)

        assert result.group_performances == group_performances
        assert result.n_groups == 3
