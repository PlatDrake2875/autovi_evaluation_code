"""Tests for robust aggregation methods."""

import numpy as np
import pytest

from src.robustness import CoordinateMedianAggregator


class TestCoordinateMedianAggregator:
    """Tests for CoordinateMedianAggregator."""

    def test_output_shape(self):
        """Test that output shape matches expected dimensions."""
        aggregator = CoordinateMedianAggregator(num_samples_per_client=50)

        # Create 5 clients with different sizes
        client_updates = [
            np.random.randn(100, 64),  # Client 0: 100 samples, 64 features
            np.random.randn(80, 64),   # Client 1: 80 samples
            np.random.randn(120, 64),  # Client 2: 120 samples
            np.random.randn(90, 64),   # Client 3: 90 samples
            np.random.randn(110, 64),  # Client 4: 110 samples
        ]

        result, stats = aggregator.aggregate(client_updates)

        # Output should be [num_samples_per_client, feature_dim]
        assert result.shape == (50, 64)
        assert stats["num_clients"] == 5
        assert stats["samples_per_client"] == 50

    def test_robustness_to_outliers(self):
        """Test that median is robust to outlier clients."""
        aggregator = CoordinateMedianAggregator(num_samples_per_client=100, seed=42)

        # Create 5 honest clients with similar values
        rng = np.random.default_rng(42)
        honest_value = 1.0
        client_updates = [
            rng.normal(honest_value, 0.1, size=(100, 10))
            for _ in range(5)
        ]

        # Add 2 malicious clients with extreme values (40% Byzantine)
        malicious_value = 1000.0
        client_updates.append(np.full((100, 10), malicious_value))
        client_updates.append(np.full((100, 10), -malicious_value))

        result, stats = aggregator.aggregate(client_updates)

        # Median should be close to honest value, not affected by outliers
        assert np.allclose(result.mean(), honest_value, atol=0.5)
        assert stats["num_clients"] == 7

    def test_single_client(self):
        """Test aggregation with a single client."""
        aggregator = CoordinateMedianAggregator(num_samples_per_client=50)

        client_updates = [np.random.randn(100, 32)]
        result, stats = aggregator.aggregate(client_updates)

        assert result.shape == (50, 32)
        assert stats["num_clients"] == 1

    def test_small_client_with_replacement(self):
        """Test that small clients are sampled with replacement."""
        aggregator = CoordinateMedianAggregator(num_samples_per_client=100)

        # Client has fewer samples than requested
        client_updates = [
            np.random.randn(20, 16),  # Only 20 samples
            np.random.randn(150, 16),  # Normal size
        ]

        result, stats = aggregator.aggregate(client_updates)

        # Should still work, sampling with replacement
        assert result.shape == (100, 16)
        assert stats["client_sizes"] == [20, 150]

    def test_empty_input_raises(self):
        """Test that empty input raises ValueError."""
        aggregator = CoordinateMedianAggregator()

        with pytest.raises(ValueError, match="No client updates provided"):
            aggregator.aggregate([])

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        client_updates = [np.random.randn(100, 32) for _ in range(3)]

        agg1 = CoordinateMedianAggregator(seed=123)
        agg2 = CoordinateMedianAggregator(seed=123)

        result1, _ = agg1.aggregate(client_updates)
        result2, _ = agg2.aggregate(client_updates)

        np.testing.assert_array_equal(result1, result2)

    def test_repr(self):
        """Test string representation."""
        aggregator = CoordinateMedianAggregator(num_samples_per_client=200, seed=99)
        repr_str = repr(aggregator)

        assert "CoordinateMedianAggregator" in repr_str
        assert "200" in repr_str
        assert "99" in repr_str
