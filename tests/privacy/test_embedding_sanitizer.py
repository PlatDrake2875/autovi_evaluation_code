"""Tests for EmbeddingSanitizer and DPConfig classes."""

import numpy as np
import pytest

from src.privacy import EmbeddingSanitizer, DPConfig


class TestDPConfig:
    """Tests for DPConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DPConfig()
        assert config.enabled is False
        assert config.epsilon == 1.0
        assert config.delta == 1e-5
        assert config.clipping_norm == 1.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DPConfig(enabled=True, epsilon=5.0, delta=1e-6, clipping_norm=2.0)
        assert config.enabled is True
        assert config.epsilon == 5.0
        assert config.delta == 1e-6
        assert config.clipping_norm == 2.0

    def test_validation_when_enabled(self):
        """Test validation only occurs when enabled."""
        # Should not raise when disabled
        config = DPConfig(enabled=False, epsilon=0.001)  # Invalid but disabled
        assert config.epsilon == 0.001

    def test_epsilon_validation(self):
        """Test epsilon range validation when enabled."""
        with pytest.raises(ValueError, match="Epsilon must be in"):
            DPConfig(enabled=True, epsilon=0.05)
        with pytest.raises(ValueError, match="Epsilon must be in"):
            DPConfig(enabled=True, epsilon=15.0)

    def test_delta_validation(self):
        """Test delta range validation when enabled."""
        with pytest.raises(ValueError, match="Delta must be in"):
            DPConfig(enabled=True, delta=0)
        with pytest.raises(ValueError, match="Delta must be in"):
            DPConfig(enabled=True, delta=1.0)

    def test_clipping_norm_validation(self):
        """Test clipping norm validation when enabled."""
        with pytest.raises(ValueError, match="Clipping norm must be positive"):
            DPConfig(enabled=True, clipping_norm=0)
        with pytest.raises(ValueError, match="Clipping norm must be positive"):
            DPConfig(enabled=True, clipping_norm=-1.0)


class TestEmbeddingSanitizerInit:
    """Tests for EmbeddingSanitizer initialization."""

    def test_init_disabled(self):
        """Test initialization with DP disabled."""
        config = DPConfig(enabled=False)
        sanitizer = EmbeddingSanitizer(config)
        assert sanitizer.mechanism is None
        assert sanitizer.config.enabled is False

    def test_init_enabled(self):
        """Test initialization with DP enabled."""
        config = DPConfig(enabled=True, epsilon=1.0, clipping_norm=1.0)
        sanitizer = EmbeddingSanitizer(config)
        assert sanitizer.mechanism is not None
        assert sanitizer.config.enabled is True


class TestClipping:
    """Tests for L2 norm clipping."""

    def test_clipping_bounds_norm(self):
        """Verify clipping bounds all norms to clipping_norm."""
        config = DPConfig(enabled=True, epsilon=1.0, clipping_norm=1.0)
        sanitizer = EmbeddingSanitizer(config)

        # Create embeddings with large norms
        embeddings = np.random.randn(100, 50) * 10  # Large norms
        clipped, _, _, _ = sanitizer.clip_embeddings(embeddings)

        norms = np.linalg.norm(clipped, axis=1)
        assert np.all(norms <= 1.0 + 1e-6)

    def test_clipping_preserves_small_norms(self):
        """Verify embeddings with small norms are not modified."""
        config = DPConfig(enabled=True, epsilon=1.0, clipping_norm=10.0)
        sanitizer = EmbeddingSanitizer(config)

        # Create embeddings with small norms (all < 10)
        embeddings = np.random.randn(100, 50) * 0.1
        original_norms = np.linalg.norm(embeddings, axis=1)
        clipped, num_clipped, _, _ = sanitizer.clip_embeddings(embeddings)
        clipped_norms = np.linalg.norm(clipped, axis=1)

        # No embeddings should be clipped
        assert num_clipped == 0
        np.testing.assert_array_almost_equal(original_norms, clipped_norms)

    def test_clipping_preserves_direction(self):
        """Verify clipping preserves embedding direction."""
        config = DPConfig(enabled=True, epsilon=1.0, clipping_norm=1.0)
        sanitizer = EmbeddingSanitizer(config)

        # Create a single embedding with known direction
        embedding = np.array([[3.0, 4.0]])  # Norm = 5, direction = (0.6, 0.8)
        clipped, _, _, _ = sanitizer.clip_embeddings(embedding)

        # Check direction is preserved
        expected_direction = np.array([0.6, 0.8])
        clipped_normalized = clipped[0] / np.linalg.norm(clipped[0])
        np.testing.assert_array_almost_equal(clipped_normalized, expected_direction)

    def test_clipping_statistics(self):
        """Verify clipping returns correct statistics."""
        config = DPConfig(enabled=True, epsilon=1.0, clipping_norm=1.0)
        sanitizer = EmbeddingSanitizer(config)

        # Create 50 large embeddings and 50 small embeddings
        large = np.random.randn(50, 50) * 10
        small = np.random.randn(50, 50) * 0.01
        embeddings = np.vstack([large, small])

        clipped, num_clipped, avg_before, avg_after = sanitizer.clip_embeddings(embeddings)

        assert num_clipped == 50  # Only large embeddings clipped
        assert avg_before > avg_after  # Average norm should decrease


class TestSanitization:
    """Tests for full sanitization pipeline."""

    def test_sanitize_disabled_returns_original(self):
        """Verify sanitization returns original when disabled."""
        config = DPConfig(enabled=False)
        sanitizer = EmbeddingSanitizer(config)

        embeddings = np.random.randn(100, 50)
        sanitized = sanitizer.sanitize(embeddings, seed=42)

        np.testing.assert_array_equal(embeddings, sanitized)

    def test_sanitize_changes_embeddings(self):
        """Verify sanitization modifies embeddings when enabled."""
        config = DPConfig(enabled=True, epsilon=1.0, clipping_norm=1.0)
        sanitizer = EmbeddingSanitizer(config)

        embeddings = np.random.randn(100, 50)
        sanitized = sanitizer.sanitize(embeddings, seed=42)

        assert not np.allclose(embeddings, sanitized)

    def test_sanitize_seed_reproducibility(self):
        """Verify sanitization is reproducible with same seed."""
        config = DPConfig(enabled=True, epsilon=1.0, clipping_norm=1.0)
        sanitizer = EmbeddingSanitizer(config)

        embeddings = np.random.randn(100, 50)
        sanitized1 = sanitizer.sanitize(embeddings, seed=42)

        # Create new sanitizer to reset state
        sanitizer2 = EmbeddingSanitizer(config)
        sanitized2 = sanitizer2.sanitize(embeddings, seed=42)

        np.testing.assert_array_equal(sanitized1, sanitized2)

    def test_sanitize_updates_stats(self):
        """Verify sanitization updates statistics."""
        config = DPConfig(enabled=True, epsilon=1.0, clipping_norm=1.0)
        sanitizer = EmbeddingSanitizer(config)

        embeddings = np.random.randn(100, 50)
        sanitizer.sanitize(embeddings, seed=42)

        stats = sanitizer.get_stats()
        assert stats["num_sanitizations"] == 1
        assert stats["total_embeddings_processed"] == 100
        assert "dp_epsilon" in stats
        assert "sigma" in stats


class TestPrivacySpent:
    """Tests for privacy budget tracking."""

    def test_privacy_spent_disabled(self):
        """Verify privacy spent is zero when disabled."""
        config = DPConfig(enabled=False)
        sanitizer = EmbeddingSanitizer(config)

        epsilon, delta = sanitizer.get_privacy_spent()
        assert epsilon == 0.0
        assert delta == 0.0

    def test_privacy_spent_enabled(self):
        """Verify privacy spent returns config values when enabled."""
        config = DPConfig(enabled=True, epsilon=5.0, delta=1e-6)
        sanitizer = EmbeddingSanitizer(config)

        epsilon, delta = sanitizer.get_privacy_spent()
        assert epsilon == 5.0
        assert delta == 1e-6


class TestStats:
    """Tests for statistics tracking."""

    def test_initial_stats(self):
        """Verify initial statistics are zeroed."""
        config = DPConfig(enabled=True, epsilon=1.0, clipping_norm=1.0)
        sanitizer = EmbeddingSanitizer(config)

        stats = sanitizer.get_stats()
        assert stats["num_sanitizations"] == 0
        assert stats["total_embeddings_processed"] == 0

    def test_stats_accumulate(self):
        """Verify statistics accumulate across sanitizations."""
        config = DPConfig(enabled=True, epsilon=1.0, clipping_norm=1.0)
        sanitizer = EmbeddingSanitizer(config)

        embeddings1 = np.random.randn(100, 50)
        embeddings2 = np.random.randn(200, 50)

        sanitizer.sanitize(embeddings1, seed=42)
        sanitizer.sanitize(embeddings2, seed=43)

        stats = sanitizer.get_stats()
        assert stats["num_sanitizations"] == 2
        assert stats["total_embeddings_processed"] == 300

    def test_stats_copy_returned(self):
        """Verify get_stats returns a copy."""
        config = DPConfig(enabled=True, epsilon=1.0, clipping_norm=1.0)
        sanitizer = EmbeddingSanitizer(config)

        stats = sanitizer.get_stats()
        stats["num_sanitizations"] = 999

        # Original should be unchanged
        assert sanitizer.get_stats()["num_sanitizations"] == 0


class TestRepr:
    """Tests for string representation."""

    def test_repr_disabled(self):
        """Test repr shows disabled status."""
        config = DPConfig(enabled=False)
        sanitizer = EmbeddingSanitizer(config)
        assert "disabled" in repr(sanitizer)

    def test_repr_enabled(self):
        """Test repr shows enabled status and parameters."""
        config = DPConfig(enabled=True, epsilon=5.0, clipping_norm=2.0)
        sanitizer = EmbeddingSanitizer(config)
        repr_str = repr(sanitizer)
        assert "enabled" in repr_str
        assert "epsilon=5.0" in repr_str
        assert "clipping_norm=2.0" in repr_str
